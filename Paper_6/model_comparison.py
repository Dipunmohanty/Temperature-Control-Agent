import numpy as np
import matplotlib.pyplot as plt
import time

# Optional: only if you want LSTM
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)

# -----------------------------
# 1. Dataset Generator
# -----------------------------
# def generate_data(n=500, complexity=1):
#     t = np.arange(n)

#     outside = 30 + 5*np.sin(2*np.pi*t/50)

#     if complexity >= 2:
#         outside += 3*np.sin(2*np.pi*t/10)

#     if complexity >= 3:
#         outside += np.random.normal(0, 1, n)

#     indoor = []
#     temp = 25

#     for i in range(n):
#         temp = temp + 0.05*(outside[i] - temp)
#         if complexity >= 3:
#             temp += np.random.normal(0, 0.5)
#         indoor.append(temp)

#     return np.array(indoor)
def generate_data(n=500, complexity=0):
    t = np.arange(n)

    # Base smooth pattern
    outside = 30 + 3*np.sin(2*np.pi*t/80)

    if complexity >= 1:
        # Add second wave (medium complexity)
        outside += 2*np.sin(2*np.pi*t/20)

    if complexity >= 2:
        # Add sudden spikes (steep changes)
        spikes = np.zeros(n)
        spike_indices = np.random.choice(n, size=10, replace=False)
        spikes[spike_indices] = np.random.uniform(-5, 5, size=10)
        outside += spikes

    if complexity >= 3:
        # Add rapid fluctuations + noise
        outside += 2*np.sin(2*np.pi*t/5)  # fast oscillation
        outside += np.random.normal(0, 1.5, n)  # noise

    # Indoor temperature (thermal inertia)
    indoor = []
    temp = 25

    for i in range(n):
        temp = temp + 0.05*(outside[i] - temp)
        indoor.append(temp)

    return np.array(indoor), outside

# -----------------------------
# 2. Prepare Data
# -----------------------------
def prepare_data(series, window=5):
    X, y = [], []
    for i in range(len(series)-window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

# -----------------------------
# 3. Lightweight Model
# -----------------------------
def train_lightweight(X, y, epochs=50):
    Xn = (X - X.mean()) / X.std()
    yn = (y - y.mean()) / y.std()

    w = np.random.randn(X.shape[1])
    b = 0

    lr = 0.01

    start = time.time()

    for _ in range(epochs):
        for i in range(len(Xn)):
            pred = np.dot(w, Xn[i]) + b
            err = pred - yn[i]

            w -= lr * 2 * err * Xn[i]
            b -= lr * 2 * err

    duration = time.time() - start

    preds = np.dot(Xn, w) + b
    preds = preds * y.std() + y.mean()

    mse = np.mean((preds - y)**2)
    rmse = np.sqrt(mse)

    return preds, mse, rmse, duration

# -----------------------------
# 4. LSTM Model
# -----------------------------
def train_lstm(X, y):
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = keras.Sequential([
        keras.layers.LSTM(32, input_shape=(X.shape[1],1)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    start = time.time()

    history = model.fit(X, y, epochs=20, verbose=0)

    duration = time.time() - start

    preds = model.predict(X, verbose=0).flatten()

    mse = np.mean((preds - y)**2)
    rmse = np.sqrt(mse)

    return preds, mse, rmse, duration

plt.figure(figsize=(10,6))

for c in range(4):
    indoor, outside = generate_data(complexity=c)
    plt.plot(outside[:200], label=f"Complexity {c}")

plt.title("Dataset Complexity Comparison (Outside Temperature)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 5. Run Experiment
# -----------------------------
complexities = [1, 2, 3]

results = []

for c in complexities:
    print(f"\n--- Complexity Level {c} ---")

    indoor, outside = generate_data(complexity=c)
    X, y = prepare_data(indoor)

    # Lightweight model
    lw_preds, lw_mse, lw_rmse, lw_time = train_lightweight(X, y)

    # LSTM model
    lstm_preds, lstm_mse, lstm_rmse, lstm_time = train_lstm(X, y)

    results.append((lw_mse, lstm_mse, lw_rmse, lstm_rmse, lw_time, lstm_time))

    print(f"Lightweight → MSE: {lw_mse:.4f}, RMSE: {lw_rmse:.4f}, Time: {lw_time:.2f}s")
    print(f"LSTM        → MSE: {lstm_mse:.4f}, RMSE: {lstm_rmse:.4f}, Time: {lstm_time:.2f}s")

# -----------------------------
# 6. Plot Comparison
# -----------------------------
lw_mse = [r[0] for r in results]
lstm_mse = [r[1] for r in results]

lw_time = [r[4] for r in results]
lstm_time = [r[5] for r in results]

plt.figure()
plt.plot(complexities, lw_mse, label="Lightweight MSE")
plt.plot(complexities, lstm_mse, label="LSTM MSE")
plt.title("Model Error vs Complexity")
plt.xlabel("Complexity Level")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(complexities, lw_time, label="Lightweight Time")
plt.plot(complexities, lstm_time, label="LSTM Time")
plt.title("Training Time vs Complexity")
plt.xlabel("Complexity Level")
plt.ylabel("Seconds")
plt.legend()
plt.grid()
plt.show()