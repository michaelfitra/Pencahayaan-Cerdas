import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import paho.mqtt.client as mqtt
from queue import Queue
import json
import torch
import torch.cuda.amp
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading

class MQTTHandler:
    def __init__(self, broker="localhost", port=1883, topic="room/occupancy"):
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.broker = broker
        self.port = port
        self.topic = topic
        self.message_queue = Queue()
        self.connected = False

        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish

    def connect(self):
        """Attempt to connect to the MQTT broker once"""
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            time.sleep(1)
            if self.connected:
                print(
                    f"Successfully connected to MQTT broker at {self.broker}:{self.port}")
                return True
            else:
                print("Failed to connect: Broker did not respond")
                return False
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client receives a CONNACK response from the server"""
        if rc == 0:
            self.connected = True
            print("Successfully connected to MQTT broker")
        else:
            error_messages = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized"
            }
            print(
                f"Failed to connect: {error_messages.get(rc, f'Unknown error code {rc}')}")
            self.connected = False

    def on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the server"""
        self.connected = False
        if rc != 0:
            print(f"Unexpected disconnection. RC: {rc}")

    def on_publish(self, client, userdata, mid):
        """Callback for when a message is published"""
        print(f"Message {mid} published successfully")

    def publish_count(self, count):
        """Publish people count with error handling"""
        if not self.connected:
            if not self.connect():
                print("Cannot publish: Not connected to broker")
                return False

        try:
            payload = json.dumps(
                {"people_count": count, "timestamp": time.time()})
            result = self.client.publish(self.topic, payload, qos=1)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(
                    f"Failed to publish message: {mqtt.error_string(result.rc)}")
                return False
            return True
        except Exception as e:
            print(f"Error publishing message: {e}")
            return False

    def disconnect(self):
        """Safely disconnect from the broker"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            print("Successfully disconnected from MQTT broker")
        except Exception as e:
            print(f"Error during disconnect: {e}")

class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_objects = {}
        self.max_distance = 50
        self.max_frames_missing = 5

    def update(self, detections):
        # Return empty tracker_id if no detections
        if len(detections.xyxy) == 0:
            detections.tracker_id = np.array([])
            return detections

        if not hasattr(detections, 'xyxy'):
            return detections

        current_boxes = detections.xyxy
        current_centers = np.array([[
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        ] for box in current_boxes])

        new_tracked_objects = {}
        used_detections = set()

        for track_id, track_info in self.tracked_objects.items():
            if track_info['frames_missing'] > self.max_frames_missing:
                continue

            track_center = track_info['center']
            distances = np.linalg.norm(current_centers - track_center, axis=1)

            if distances.size > 0:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]

                if min_dist < self.max_distance and min_dist_idx not in used_detections:
                    new_tracked_objects[track_id] = {
                        'center': current_centers[min_dist_idx],
                        'frames_missing': 0
                    }
                    used_detections.add(min_dist_idx)
                else:
                    track_info['frames_missing'] += 1
                    if track_info['frames_missing'] <= self.max_frames_missing:
                        new_tracked_objects[track_id] = track_info

        for i in range(len(current_centers)):
            if i not in used_detections:
                new_tracked_objects[self.next_id] = {
                    'center': current_centers[i],
                    'frames_missing': 0
                }
                self.next_id += 1

        self.tracked_objects = new_tracked_objects

        tracker_ids = np.array([
            track_id
            for track_id, track_info in self.tracked_objects.items()
            if track_info['frames_missing'] == 0
        ])

        detections.tracker_id = tracker_ids
        return detections

class PersonCounter:
    def __init__(self, show_line=False, mqtt_broker="localhost"):
        # Setup logging with file output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('person_counter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.fps_history = []
        self.max_fps_history = 100
        self.performance_lock = threading.Lock()
        
        # Optimisasi CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        if self.use_cuda:
            self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.scaler = torch.cuda.amp.GradScaler()
            torch.cuda.empty_cache()
            
            # Set optimal CUDA settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.logger.info("Using CPU")

        # Thread pool untuk operasi non-GPU
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Buffer untuk batch processing
        self.frame_buffer = []
        self.batch_size = 4
        self.frame_lock = threading.Lock()
        
        self.show_line = show_line
        self.mqtt_handler = MQTTHandler(broker=mqtt_broker)
        
        # Initialize components
        self.detect_cameras()
        self.initialize_camera()
        self.initialize_models()
        
        self.running = True
        self.last_mqtt_publish = 0
        self.mqtt_publish_interval = 1.0

    def update_performance_metrics(self, fps):
        with self.performance_lock:
            self.fps_history.append(fps)
            if len(self.fps_history) > self.max_fps_history:
                self.fps_history.pop(0)
            
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.logger.debug(f"Average FPS: {avg_fps:.2f}")

    def initialize_models(self):
        try:
            self.logger.info("Loading YOLO model...")
            self.model = YOLO('yolov8n.pt')
            
            if self.use_cuda:
                self.model.to(self.device)
                self.model.model.eval()
                # Mengoptimalkan model untuk inferensi
                try:
                    self.model = torch.jit.script(self.model.model) if hasattr(self.model, 'model') else self.model
                except Exception as e:
                    self.logger.warning(f"JIT compilation failed, using standard model: {e}")
            else:
                self.logger.info("Running on CPU - optimizations disabled")
                self.model.model.eval() if hasattr(self.model, 'model') else None
            
            torch.set_grad_enabled(False)
            
            self.tracker = SimpleTracker()
            self.line_y = self.frame_height // 2
            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            raise

    def detect_cameras(self):
        self.logger.info("Searching for available cameras...")
        available_cameras = []

        for index in range(10):
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(index)
                        self.logger.info(f"Camera found at index {index}")
                cap.release()
            except Exception as e:
                self.logger.debug(f"Error checking camera {index}: {e}")

        if not available_cameras:
            raise ValueError("No cameras detected")
        
        self.camera_index = available_cameras[0]
        self.logger.info(f"Selected camera index: {self.camera_index}")
        return

    def initialize_camera(self):
        try:
            self.logger.info(f"Attempting to connect to camera {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.logger.info("Trying DSHOW backend...")
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            time.sleep(2)

            if not self.cap.isOpened():
                raise ValueError("Failed to open camera")

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Verify camera initialization
            for _ in range(5):
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    self.frame_height, self.frame_width = test_frame.shape[:2]
                    self.logger.info(
                        f"Camera initialized successfully. Resolution: {self.frame_width}x{self.frame_height}")
                    return
                time.sleep(0.5)

            raise ValueError("Could not get valid frame from camera")

        except Exception as e:
            if hasattr(self, 'cap'):
                self.cap.release()
            raise ValueError(f"Camera initialization error: {str(e)}")

    def process_frame(self, frame):
        try:
            # Batch processing
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) < self.batch_size:
                return frame

            # Process batch
            with torch.cuda.amp.autocast() if self.use_cuda else nullcontext():
                batch_frames = torch.stack([torch.from_numpy(f).to(self.device) 
                                         for f in self.frame_buffer])
                
                with torch.no_grad():
                    results = self.model(batch_frames, verbose=False)

            # Clear buffer
            self.frame_buffer.clear()
            
            # Process current frame
            result = results[0]
            
            # Async non-GPU operations
            future_annotations = self.executor.submit(self._prepare_annotations, 
                                                    result, frame)
            
            # While waiting for annotations, prepare next frame
            if self.use_cuda:
                torch.cuda.current_stream().synchronize()
            
            # Get annotations
            return future_annotations.result()

        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return frame

    def _prepare_annotations(self, result, frame):
        try:
            if self.use_cuda:
                boxes = result.boxes.cpu()
            else:
                boxes = result.boxes

            detections = sv.Detections(
                xyxy=boxes.xyxy.numpy(),
                confidence=boxes.conf.numpy(),
                class_id=boxes.cls.numpy().astype(np.int32)
            )

            mask = detections.class_id == 0
            detections = detections[mask]
            detections = self.tracker.update(detections)

            return self._draw_annotations(frame, detections)

        except Exception as e:
            self.logger.error(f"Annotation error: {str(e)}")
            return frame

    def _draw_annotations(self, frame, detections):
        try:
            annotated_frame = frame.copy()
            
            # Prepare batch text rendering
            texts = []
            positions = []
            boxes = []
            
            if len(detections.xyxy) > 0:
                # Collect all drawing operations
                for bbox, conf, tracker_id in zip(detections.xyxy, 
                                                detections.confidence, 
                                                detections.tracker_id):
                    x1, y1, x2, y2 = map(int, bbox)
                    boxes.append(((x1, y1), (x2, y2)))
                    texts.append(f"ID:{int(tracker_id)} {conf:.2f}")
                    positions.append((x1, y1 - 10))

                # Batch draw rectangles
                for (start_point, end_point) in boxes:
                    cv2.rectangle(annotated_frame, start_point, end_point, (0, 255, 0), 2)

                # Batch draw text backgrounds and text
                for text, pos in zip(texts, positions):
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    x, y = pos
                    cv2.rectangle(annotated_frame, 
                                (x, y - text_height - 10),
                                (x + text_width, y),
                                (0, 0, 0), -1)
                    cv2.putText(annotated_frame, text, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add counting line
            if self.show_line:
                cv2.line(annotated_frame, (0, self.line_y),
                        (self.frame_width, self.line_y), (0, 255, 0), 2)

            # Add people count and FPS
            people_count = len(detections.xyxy)
            self._draw_stats(annotated_frame, people_count)

            return annotated_frame

        except Exception as e:
            self.logger.error(f"Drawing error: {str(e)}")
            return frame

    def _draw_stats(self, frame, people_count):
        # Draw people count
        count_text = f"Jumlah orang: {people_count}"
        self._draw_text_with_background(frame, count_text, (10, 35))

        # Draw FPS
        if hasattr(self, 'fps_history') and self.fps_history:
            avg_fps = sum(self.fps_history[-10:]) / min(len(self.fps_history), 10)
            fps_text = f"FPS: {avg_fps:.1f}"
            self._draw_text_with_background(frame, fps_text, (10, 75))

    def _draw_text_with_background(self, frame, text, position, font_scale=1.0):
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        x, y = position
        cv2.rectangle(frame, (x-5, y-text_height-5),
                     (x+text_width+5, y+5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    def run(self):
        self.logger.info("Starting person detection...")
        retry_count = 0
        max_retries = 3
        self.start_time = time.time()
        last_fps_update = time.time()
        fps_update_interval = 0.5  # Update FPS setiap 0.5 detik

        try:
            cv2.namedWindow('Person Counter', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Person Counter', self.frame_width, self.frame_height)

            while self.running:
                if cv2.getWindowProperty('Person Counter', cv2.WND_PROP_VISIBLE) < 1:
                    break

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.logger.warning("Failed to grab frame.")
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error("Too many failed frame grabs. Exiting...")
                        break
                    time.sleep(1)
                    continue

                retry_count = 0
                processed_frame = self.process_frame(frame)

                # Update FPS
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    fps = 1 / (current_time - self.start_time)
                    self.update_performance_metrics(fps)
                    last_fps_update = current_time
                self.start_time = current_time

                try:
                    cv2.imshow('Person Counter', processed_frame)
                except Exception as e:
                    self.logger.error(f"Error displaying frame: {str(e)}")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Program stopped by user")
                    break

        except KeyboardInterrupt:
            self.logger.info("Program interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.logger.info("Cleaning up resources...")
        try:
            self.executor.shutdown(wait=True)
            if self.use_cuda:
                torch.cuda.empty_cache()
            if hasattr(self, 'cap'):
                self.cap.release()
            self.mqtt_handler.disconnect()
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
        finally:
            self.running = False

if __name__ == "__main__":
    try:
        counter = PersonCounter(show_line=False, mqtt_broker="192.168.94.184")
        counter.run()
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")
    finally:
        if 'counter' in locals():
            counter.cleanup()
        exit()
