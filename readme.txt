SISTEM PENCAHAYAAN CERDAS BERBASIS JUMLAH ORANG
==============================================

Dibuat oleh: Mikel
Untuk masukan atau saran, silakan hubungi:
- WhatsApp: +62851 5888 9868
- GitHub: github.com/michaelfitra

Deskripsi Proyek
---------------
Sistem ini mengintegrasikan deteksi jumlah orang menggunakan kamera dengan kontrol pencahayaan otomatis menggunakan ESP32. Sistem terdiri dari dua bagian utama:
1. Program deteksi orang (PersonCounter.py)
2. Kontroler pencahayaan (ESP32)

Kebutuhan Sistem
---------------
1. Python 3.10.11
2. ESP32 DevKit v1
3. Sensor cahaya (LDR)
4. LED
5. Mosquitto MQTT Broker
6. Kamera webcam

Dependensi Python
----------------
- OpenCV (cv2) 4.10.0
- NumPy 2.2.0
- YOLO (ultralytics 8.3.52)
- supervision 0.25.1
- paho-mqtt 1.6.1

Dependensi Arduino
-----------------
- PubSubClient
- ArduinoJson
- WiFi (built-in)

Instalasi Mosquitto
------------------
1. Download Mosquitto dari https://mosquitto.org/download/
2. Install Mosquitto dan pastikan servis berjalan
3. Jalankan "InisiasiMosquitto.bat" untuk memulai broker MQTT
4. Pastikan port 1883 tersedia

Konfigurasi
-----------
1. ESP32:
   - Sesuaikan SSID dan password WiFi
   - Sesuaikan IP broker MQTT
   - Upload kode ke ESP32

2. PersonCounter.py:
   - Sesuaikan IP broker MQTT pada inisialisasi PersonCounter
   - Pastikan kamera terdeteksi

3. Koneksi Hardware:
   - LED -> Pin 2
   - LDR -> Pin 34
   - Gunakan resistor pull-down 10k ohm untuk LDR
   - Gunakan resistor 220 ohm untuk LED

Cara Kerja
---------
1. PersonCounter.py:
   - Mendeteksi jumlah orang menggunakan YOLO
   - Mengirim data ke broker MQTT
   - Menampilkan visualisasi deteksi

2. ESP32:
   - Menerima data jumlah orang dari MQTT
   - Membaca nilai sensor cahaya
   - Menggunakan logika fuzzy untuk menentukan tingkat pencahayaan
   - Mengontrol LED sesuai hasil perhitungan

3. Logika Fuzzy:
   - Input: jumlah orang dan tingkat cahaya
   - Output: intensitas LED (PWM)
   - Rules disesuaikan untuk optimasi pencahayaan

Menjalankan Sistem
----------------
1. Jalankan InisiasiMosquitto.bat
2. Upload kode ke ESP32
3. Jalankan PersonCounter.py
4. Sistem akan mulai mendeteksi dan mengontrol pencahayaan

Troubleshooting
-------------
1. Koneksi MQTT gagal:
   - Periksa IP broker
   - Pastikan firewall tidak memblokir port 1883
   - Cek status Mosquitto

2. Kamera tidak terdeteksi:
   - Periksa indeks kamera
   - Pastikan tidak ada aplikasi lain yang menggunakan kamera

3. ESP32 tidak merespon:
   - Periksa koneksi WiFi
   - Reset ESP32
   - Verifikasi wiring hardware

Catatan Penting
-------------
- Jangan tutup InisiasiMosquitto.bat menggunakan tombol X
- Gunakan Ctrl+C untuk menghentikan program dengan benar
- Kalibrasi sensor cahaya mungkin diperlukan sesuai kondisi ruangan
- Program dapat dimodifikasi untuk menggunakan threshold berbeda
