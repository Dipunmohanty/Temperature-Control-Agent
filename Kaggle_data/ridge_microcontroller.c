#include <Wire.h>
#include <DHT.h>
#include <Adafruit_BMP085.h>
#include <LiquidCrystal_I2C.h>
#include <math.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

#define DHTPIN 2
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

Adafruit_BMP085 bmp;

#define FAN_PIN 7

float temp_lag1 = 0;
float temp_lag2 = 0;
bool first = true;

float predict(float x[]) {
    float y = -19.327715f;

    y += 0.383699f * x[0]; // Temp
    y += 0.286057f * x[1]; // temp_lag1
    y += 0.264205f * x[2]; // temp_lag2
    y += 0.009328f * x[3]; // Humid
    y += 0.019415f * x[4]; // Pressure

    return y;
}

void setup() {
    Serial.begin(9600);

    dht.begin();
    bmp.begin();

    lcd.init();
    lcd.backlight();

    pinMode(FAN_PIN, OUTPUT);

    Serial.println("Smart Climate System Ready");
}

void loop() {

    float t = millis() / 1000.0;

    float temp = 25 + 5 * sin(t / 60.0);       // daily cycle
    float humid = 50 + 20 * cos(t / 90.0);     // humidity variation
    float pressure = 1010 + 5 * sin(t / 120.0);

    if (first) {
        temp_lag1 = temp;
        temp_lag2 = temp;
        first = false;
    }

    float x[5];

    x[0] = temp;
    x[1] = temp_lag1;
    x[2] = temp_lag2;
    x[3] = humid;
    x[4] = pressure;

    float predicted = predict(x);

    // ===== FAN CONTROL =====
    if (predicted > 28) {
        digitalWrite(FAN_PIN, HIGH);  // ON
    } else {
        digitalWrite(FAN_PIN, LOW);   // OFF
    }

    // ===== SERIAL OUTPUT =====
    Serial.print("Temp: "); Serial.print(temp);
    Serial.print(" | Humid: "); Serial.print(humid);
    Serial.print(" | Pressure: "); Serial.print(pressure);
    Serial.print(" | Pred: "); Serial.println(predicted);

    // ===== LCD DISPLAY =====
    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("T:");
    lcd.print(temp,1);
    lcd.print(" P:");
    lcd.print(predicted,1);

    lcd.setCursor(0, 1);
    lcd.print("H:");
    lcd.print(humid,0);
    lcd.print(" Fan:");
    lcd.print(predicted > 28 ? "ON" : "OFF");

    // ===== UPDATE LAG =====
    temp_lag2 = temp_lag1;
    temp_lag1 = temp;

    delay(2000);
}