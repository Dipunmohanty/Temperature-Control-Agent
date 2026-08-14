// Auto-generated Ridge model

float predict(float x[]) {
    float y = -19.327715f;

    y += 0.383699f * x[0]; // Temp
    y += 0.286057f * x[1]; // temp_lag1
    y += 0.264205f * x[2]; // temp_lag2
    y += 0.009328f * x[3]; // Humid
    y += 0.019415f * x[4]; // Pressure

    return y;
}
