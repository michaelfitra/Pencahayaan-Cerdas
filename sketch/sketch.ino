#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
// #include <ESP32PWM.h>

// Konfigurasi WiFi
const char* ssid = "☌⌰⟟⟒⏚⊬";
const char* password = "acumalaka";

// Konfigurasi MQTT
const char* mqtt_server = "192.168.250.184"; // IP lokal komputer
const int mqtt_port = 1883;
const char* mqtt_topic = "room/occupancy";

// Konfigurasi Pin
const int LED_PIN = 2;        // Pin LED bawaan ESP32
const int PHOTO_PIN = 34;     // Pin analog untuk sensor cahaya
const int PWM_CHANNEL = 0;    // Saluran PWM
const int PWM_FREQ = 5000;    // Frekuensi PWM
const int PWM_RESOLUTION = 8; // Resolusi PWM (8 bit - rentang 0-255)

// Objek-objek
WiFiClient espClient;
PubSubClient mqtt(espClient);

// Variabel global
int peopleCount = 0;
float lightLevel = 0;
unsigned long lastSensorRead = 0;
const int SENSOR_READ_INTERVAL = 1000; // Interval pembacaan sensor (ms)

// Fungsi keanggotaan fuzzy untuk jumlah orang
float peopleEmpty(int x) {
  if (x <= 0) return 1;
  if (x >= 9) return 0;
  return (9.0 - x) / 9.0;
}

float peopleMedium(int x) {
  if (x <= 0 || x >= 18) return 0;
  if (x >= 6 && x <= 12) return 1;
  if (x < 6) return (x - 0.0) / 6.0;
  return (18.0 - x) / 6.0;
}

float peopleFull(int x) {
  if (x <= 9) return 0;
  if (x >= 18) return 1;
  return (x - 9.0) / 9.0;
}

// Fungsi keanggotaan fuzzy untuk tingkat cahaya (rentang 0-4095)
// Revisi fungsi keanggotaan untuk lebih halus
float lightDark(float x) {
  if (x >= 3900) return 1;
  if (x <= 3500) return 0;
  return (x - 3500.0) / 400.0;
}

float lightNormal(float x) {
  if (x <= 3300 || x >= 3900) return 0;
  if (x >= 3500 && x <= 3700) return 1;
  if (x < 3500) return (x - 3300.0) / 200.0;
  return (3900.0 - x) / 200.0;
}

float lightBright(float x) {
  if (x <= 2900) return 1;
  if (x >= 3300) return 0;
  return (3300.0 - x) / 400.0;
}

// Fungsi pengaturan WiFi
void setupWiFi() {
  Serial.println("Menghubungkan ke WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi terhubung");
  Serial.println("Alamat IP: ");
  Serial.println(WiFi.localIP());
}

// Fungsi callback MQTT
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, payload, length);

  if (error) {
    Serial.print("deserializeJson() gagal: ");
    Serial.println(error.c_str());
    return;
  }

  peopleCount = doc["people_count"];
  Serial.print("Jumlah orang: ");
  Serial.println(peopleCount);
}

// Fungsi reconnect MQTT
void reconnectMQTT() {
  while (!mqtt.connected()) {
    Serial.println("Mencoba koneksi MQTT...");
    String clientId = "ESP32Client-" + String(random(0xffff), HEX);

    if (mqtt.connect(clientId.c_str())) {
      Serial.println("Terhubung ke broker MQTT");
      mqtt.subscribe(mqtt_topic);
    } else {
      Serial.print("Gagal, rc=");
      Serial.print(mqtt.state());
      Serial.println(" Mencoba ulang dalam 5 detik...");
      delay(5000);
    }
  }
}

// Menghitung output fuzzy
int calculateFuzzyOutput() {
  // Menghitung derajat keanggotaan untuk jumlah orang
  float pEmpty = peopleEmpty(peopleCount);
  float pMedium = peopleMedium(peopleCount);
  float pFull = peopleFull(peopleCount);

  // Menghitung derajat keanggotaan untuk tingkat cahaya
  float lDark = lightDark(lightLevel);
  float lNormal = lightNormal(lightLevel);
  float lBright = lightBright(lightLevel);

  // Aturan dengan nilai yang lebih proporsional
  float rules[][3] = {
    {pEmpty, lDark, 180},    // Sepi & Redup -> Agak Terang
    {pEmpty, lNormal, 100},  // Sepi & Normal -> Redup
    {pEmpty, lBright, 60},   // Sepi & Terang -> Sangat Redup
    {pMedium, lDark, 255},   // Sedang & Redup -> Sangat Terang
    {pMedium, lNormal, 180}, // Sedang & Normal -> Agak Terang  
    {pMedium, lBright, 100}, // Sedang & Terang -> Redup
    {pFull, lDark, 255},     // Ramai & Redup -> Sangat Terang
    {pFull, lNormal, 230},   // Ramai & Normal -> Terang
    {pFull, lBright, 180}    // Ramai & Terang -> Agak Terang
  };

  float sumWeight = 0;
  float sumProduct = 0;

  // Menghitung rata-rata berbobot (defuzzifikasi Sugeno)
  for (int i = 0; i < 9; i++) {
    float ruleStrength = min(rules[i][0], rules[i][1]);
    sumWeight += ruleStrength;
    sumProduct += ruleStrength * rules[i][2];
  }

  if (sumWeight == 0) return 128; // Nilai default jika tidak ada aturan yang aktif
  return (int)(sumProduct / sumWeight);
}

void setup() {
  Serial.begin(115200);

  // Pengaturan LED PWM
  // ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);  // Konfigurasi PWM
  // ledcAttachPin(LED_PIN, PWM_CHANNEL);               // Menghubungkan PIN ke channel PWM
  // ledcWrite(PWM_CHANNEL, 0);                         // Inisialisasi PWM dengan nilai 0

  pinMode(LED_PIN, OUTPUT);

  // Pengaturan pin sensor cahaya
  pinMode(PHOTO_PIN, INPUT);

  // Pengaturan WiFi
  setupWiFi();

  // Pengaturan MQTT
  mqtt.setServer(mqtt_server, mqtt_port);
  mqtt.setCallback(mqttCallback);
}

void loop() {
  // Menjaga koneksi MQTT
  if (!mqtt.connected()) {
    reconnectMQTT();
  }
  mqtt.loop();

  // Membaca sensor secara berkala
  unsigned long currentMillis = millis();
  if (currentMillis - lastSensorRead >= SENSOR_READ_INTERVAL) {
    lastSensorRead = currentMillis;

    // Membaca nilai sensor cahaya (0-4095)
    lightLevel = analogRead(PHOTO_PIN);
    Serial.print("Tingkat cahaya: ");
    Serial.print(lightLevel);
    Serial.println(" nilai mentah");

    // Menghitung dan menerapkan PWM
    int pwmValue = calculateFuzzyOutput();
    analogWrite(LED_PIN, pwmValue);

    Serial.print("Output PWM: ");
    Serial.println(pwmValue);
  }

  delay(10); // Delay kecil untuk mencegah reset watchdog
}