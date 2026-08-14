import numpy as np
import pandas as pd
import tensorflow as tf
import time

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Generate Complex Data
# -----------------------------
np.random.seed(42)
t = np.arange(0, 300)

outdoor = 10 + 8*np.sin(2*np.pi*t/50) + 3*np.sin(2*np.pi*t/15) + np.random.normal(0, 1, 300)
humidity = 50 + 20*np.sin(2*np.pi*t/60) + np.random.normal(0, 5, 300)
hvac = (outdoor > 18).astype(int)

indoor = []
for i in range(300):
    if i < 5:
        indoor.append(outdoor[i])
    else:
        val = (
            0.5 * indoor[-1] +
            0.2 * outdoor[i-2] +
            0.1 * outdoor[i-4] +
            0.05 * humidity[i] +
            -2 * hvac[i] +
            0.02 * (outdoor[i]**2)/10 +
            np.random.normal(0, 0.8)
        )
        indoor.append(val)

indoor = np.array(indoor)

data = pd.DataFrame({
    "outdoor": outdoor,
    "humidity": humidity,
    "hvac": hvac,
    "indoor": indoor
})

# -----------------------------
# 2. Ridge Model
# -----------------------------
for lag in range(1, 5):
    data[f"lag{lag}"] = data["outdoor"].shift(lag)

data_ridge = data.dropna().reset_index(drop=True)

X_ridge = data_ridge.drop(columns=["indoor"])
y_ridge = data_ridge["indoor"]

split = int(0.8 * len(data_ridge))
Xr_train, Xr_test = X_ridge[:split], X_ridge[split:]
yr_train, yr_test = y_ridge[:split], y_ridge[split:]

start = time.time()
ridge = Ridge(alpha=1.0)
ridge.fit(Xr_train, yr_train)
ridge_time = time.time() - start

ridge_pred = ridge.predict(Xr_test)
ridge_rmse = np.sqrt(mean_squared_error(yr_test, ridge_pred))

# Flash usage (weights only)
ridge_flash = (ridge.coef_.nbytes + ridge.intercept_.nbytes) / 1024  # KB

# RAM usage (input + weights during inference)
ridge_ram = (Xr_test.shape[1] * 8 + ridge.coef_.nbytes) / 1024  # KB

# Inference latency
start = time.time()
_ = ridge.predict(Xr_test.iloc[0:1])
ridge_latency = (time.time() - start) * 1000

# -----------------------------
# 3. LSTM Model
# -----------------------------
data_lstm = data.dropna().reset_index(drop=True)

X = data_lstm.drop(columns=["indoor"])
y = data_lstm["indoor"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(X, y, seq_len=5):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y.values)

split = int(0.8 * len(X_seq))
Xl_train, Xl_test = X_seq[:split], X_seq[split:]
yl_train, yl_test = y_seq[:split], y_seq[split:]

start = time.time()

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xl_train.shape[1], Xl_train.shape[2])),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(Xl_train, yl_train, epochs=20, verbose=0)

lstm_time = time.time() - start

lstm_pred = lstm_model.predict(Xl_test).flatten()
lstm_rmse = np.sqrt(mean_squared_error(yl_test, lstm_pred))

# -----------------------------
# 4. Convert to TFLite
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Flash usage
lstm_flash = len(tflite_model) / 1024  # KB

# RAM usage estimation (input + activations)
input_size = np.prod(Xl_test[0].shape) * 4  # float32
hidden_units = 16
lstm_ram = (input_size + hidden_units * 4 * 4) / 1024  # KB approx

# -----------------------------
# 5. Inference Time
# -----------------------------
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample = np.array([Xl_test[0]], dtype=np.float32)

start = time.time()
interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()
lstm_latency = (time.time() - start) * 1000

# -----------------------------
# 6. Final Results
# -----------------------------
print("\n===== TinyML Benchmark =====")

print("\n--- RMSE ---")
print(f"Ridge: {ridge_rmse:.4f}")
print(f"LSTM: {lstm_rmse:.4f}")

print("\n--- Training Time ---")
print(f"Ridge: {ridge_time:.6f} sec")
print(f"LSTM: {lstm_time:.6f} sec")

print("\n--- Flash Usage ---")
print(f"Ridge: {ridge_flash:.4f} KB")
print(f"LSTM: {lstm_flash:.2f} KB")

print("\n--- RAM Usage (Estimated) ---")
print(f"Ridge: {ridge_ram:.4f} KB")
print(f"LSTM: {lstm_ram:.2f} KB")

print("\n--- Inference Time ---")
print(f"Ridge: {ridge_latency:.6f} ms")
print(f"LSTM: {lstm_latency:.4f} ms")