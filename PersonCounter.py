import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time

import paho.mqtt.client as mqtt
from queue import Queue
import json
import time

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
                print(f"Successfully connected to MQTT broker at {self.broker}:{self.port}")
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
    def __init__(self, show_line=False, mqtt_broker="localhost"): #(..., mqtt_broker="localhost"):
        self.show_line = show_line

        self.mqtt_handler = MQTTHandler(broker=mqtt_broker)
        if not self.mqtt_handler.connect():
            print("Warning: MQTT connection failed, will retry during operation")

        self.detect_cameras()
        self.initialize_camera()
        self.initialize_models()
        self.running = True  # Add flag to control the main loop
        
        self.last_mqtt_publish = 0
        self.mqtt_publish_interval = 1.0  # Publish interval in seconds

    def detect_cameras(self):
        print("Mencari kamera yang tersedia...")

        for index in range(10):
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"Kamera ditemukan pada indeks {index}")
                        cap.release()
                        self.camera_index = index
                        return
                cap.release()
            except Exception as e:
                continue

        raise ValueError("Tidak ada kamera yang terdeteksi")

    def initialize_camera(self):
        try:
            print(f"Mencoba menghubungkan ke kamera {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print("Mencoba dengan backend DSHOW...")
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            time.sleep(2)

            if not self.cap.isOpened():
                raise ValueError("Gagal membuka kamera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            for _ in range(5):
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    self.frame_height, self.frame_width = test_frame.shape[:2]
                    print(
                        f"Kamera berhasil diinisialisasi dengan resolusi: {self.frame_width}x{self.frame_height}")
                    return
                time.sleep(0.5)

            raise ValueError("Tidak bisa mendapatkan frame valid dari kamera")

        except Exception as e:
            if hasattr(self, 'cap'):
                self.cap.release()
            raise ValueError(f"Error saat inisialisasi kamera: {str(e)}")

    def initialize_models(self):
        try:
            print("Memuat model YOLO...")
            self.model = YOLO('yolov8n.pt')
            self.tracker = SimpleTracker()

            # Garis untuk visualisasi
            self.line_y = self.frame_height // 2

            print("Model berhasil dimuat")

        except Exception as e:
            raise ValueError(f"Error saat menginisialisasi model: {str(e)}")

    def process_frame(self, frame):
        try:
            results = self.model(frame, verbose=False)[0]

            # Create detections using the current format
            detections = sv.Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int)
            )

            # Filter untuk class person (ID 0)
            mask = np.array(
                [class_id == 0 for class_id in detections.class_id], dtype=bool)
            detections = detections[mask]

            # Update tracking
            detections = self.tracker.update(detections)

            # Create a copy of the frame for annotation
            annotated_frame = frame.copy()

            # Draw boxes and labels manually using cv2
            if len(detections.xyxy) > 0:
                for idx, (bbox, conf, tracker_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.tracker_id)):
                    x1, y1, x2, y2 = map(int, bbox)
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    # Draw ID and confidence
                    label = f"ID:{int(tracker_id)} {conf:.2f}"
                    # Add black background to text for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10),
                                  (x1 + text_width, y1), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw counting line only if show_line is True
            if self.show_line:
                cv2.line(annotated_frame,
                         (0, self.line_y),
                         (self.frame_width, self.line_y),
                         (0, 255, 0), 2)

            # Add people count with black background
            people_count = len(detections.xyxy)
            count_text = f"Jumlah orang: {people_count}"
            (text_width, text_height), _ = cv2.getTextSize(
                count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(annotated_frame, (10, 10),
                          (10 + text_width, 40), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                count_text,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Add FPS counter
            end_time = time.time()
            fps = 1 / (end_time - self.start_time)
            self.start_time = end_time

            fps_text = f"FPS: {fps:.1f}"
            (fps_width, _), _ = cv2.getTextSize(
                fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(annotated_frame, (10, 50),
                          (10 + fps_width, 80), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Pengiriman data ke MQTT dengan rate limiting
            current_time = time.time()
            if current_time - self.last_mqtt_publish >= self.mqtt_publish_interval:
                people_count = len(detections.xyxy)
                if self.mqtt_handler.publish_count(people_count):
                    self.last_mqtt_publish = current_time

            return annotated_frame

        except Exception as e:
            print(f"Error dalam pemrosesan frame: {str(e)}")
            return frame

    def cleanup(self):
        """Clean up resources"""
        print("Membersihkan resources...")
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            self.mqtt_handler.disconnect()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error ketika membersihkan: {e}")
        finally:
            print("Program selesai")
            self.running = False

    def run(self):
        print("Memulai deteksi orang...")
        retry_count = 0
        max_retries = 3
        self.start_time = time.time()

        try:
            # Create window first
            cv2.namedWindow('Person Counter', cv2.WINDOW_NORMAL)

            while self.running:
                # Check if window was closed
                if cv2.getWindowProperty('Person Counter', cv2.WND_PROP_VISIBLE) < 1:
                    break

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Gagal mengambil frame.")
                    retry_count += 1

                    if retry_count > max_retries:
                        print(
                            "Terlalu banyak kegagalan membaca frame. Menghentikan program...")
                        break

                    print(
                        f"Mencoba mengulang... (Percobaan {retry_count}/{max_retries})")
                    time.sleep(1)
                    continue

                retry_count = 0
                processed_frame = self.process_frame(frame)

                try:
                    cv2.imshow('Person Counter', processed_frame)
                except Exception as e:
                    print(f"Error menampilkan frame: {str(e)}")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Program dihentikan oleh pengguna")
                    break

        except KeyboardInterrupt:
            print("Program dihentikan oleh pengguna")
        except Exception as e:
            print(f"Error tak terduga: {str(e)}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    try:
        counter = PersonCounter(show_line=False, mqtt_broker="192.168.250.184") #(... , mqtt_broker="192.168.1.100")
        counter.run()
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna")
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Tekan Enter untuk keluar...")
    finally:
        if 'counter' in locals():
            counter.cleanup()
        exit()
