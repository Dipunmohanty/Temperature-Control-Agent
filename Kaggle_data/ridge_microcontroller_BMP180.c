#include <Wire.h>
#include <DHT.h>
#include <Adafruit_BMP085.h>  // Works for BMP180

#define DHTPIN 2
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);
Adafruit_BMP085 bmp;

float temp_lag1 = 0;
float temp_lag2 = 0;

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

    if (!bmp.begin()) {
        Serial.println("BMP180 not found!");
        while (1);
    }

    Serial.println("System Ready...");
}

void loop() {

    // Read sensors
    float temp = dht.readTemperature();
    float humid = dht.readHumidity();
    float pressure = bmp.readPressure() / 100.0; // Convert Pa → hPa

    // Check sensor errors
    if (isnan(temp) || isnan(humid)) {
        Serial.println("DHT sensor error!");
        delay(2000);
        return;
    }

    float x[5];

    x[0] = temp;
    x[1] = temp_lag1;
    x[2] = temp_lag2;
    x[3] = humid;
    x[4] = pressure;

    float predicted = predict(x);

    Serial.print("Temp: ");
    Serial.print(temp);
    Serial.print(" | Humid: ");
    Serial.print(humid);
    Serial.print(" | Pressure: ");
    Serial.print(pressure);
    Serial.print(" | Predicted Temp: ");
    Serial.println(predicted);

    temp_lag2 = temp_lag1;
    temp_lag1 = temp;

    delay(2000);
}