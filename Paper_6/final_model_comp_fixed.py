import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

np.random.seed(42)

# -----------------------------
# 1. Dataset Generator
# -----------------------------
def generate_data(n=500, complexity=0):
    t = np.arange(n)

    outside = 30 + 3*np.sin(2*np.pi*t/80)

    if complexity >= 1:
        outside += 2*np.sin(2*np.pi*t/20)

    if complexity >= 2:
        spikes = np.zeros(n)
        spike_idx = np.random.choice(n, size=10, replace=False)
        spikes[spike_idx] = np.random.uniform(-5, 5, 10)
        outside += spikes

    if complexity >= 3:
        outside += 2*np.sin(2*np.pi*t/5)
        outside += np.random.normal(0, 1.5, n)

    indoor = []
    temp = 25

    for i in range(n):
        temp = temp + 0.05*(outside[i] - temp)
        indoor.append(temp)

    return np.array(indoor)

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
def train_lightweight(X, y, epochs=30):
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

    return preds, duration

# -----------------------------
# 4. LSTM Model (FIXED)
# -----------------------------
def train_lstm(X, y):
    # Normalize
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()

    Xn = (X - X_mean) / X_std
    yn = (y - y_mean) / y_std

    Xn = Xn.reshape((Xn.shape[0], Xn.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xn.shape[1], 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    start = time.time()

    model.fit(Xn, yn, epochs=30, verbose=0)

    duration = time.time() - start

    preds = model.predict(Xn, verbose=0).flatten()

    # Denormalize
    preds = preds * y_std + y_mean

    return preds, duration

# -----------------------------
# 5. Experiment
# -----------------------------
complexities = [0, 1, 2, 3]

results = []
all_preds = []

for c in complexities:
    print(f"\n--- Complexity {c} ---")

    data = generate_data(complexity=c)
    X, y = prepare_data(data)

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train models
    lw_preds_train, lw_time = train_lightweight(X_train, y_train)
    lstm_preds_train, lstm_time = train_lstm(X_train, y_train)

    # Evaluate on TEST set
    lw_preds = lw_preds_train[-len(y_test):]
    lstm_preds = lstm_preds_train[-len(y_test):]

    lw_mse = np.mean((lw_preds - y_test)**2)
    lstm_mse = np.mean((lstm_preds - y_test)**2)

    results.append((lw_mse, lstm_mse, lw_time, lstm_time))
    all_preds.append((lw_preds, lstm_preds, y_test))

    print(f"Lightweight MSE: {lw_mse:.3f}")
    print(f"LSTM MSE: {lstm_mse:.3f}")

# -----------------------------
# 6. MSE Plots
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, c in enumerate(complexities):
    lw_mse, lstm_mse, _, _ = results[i]

    axes[i].bar(["Lightweight", "LSTM"], [lw_mse, lstm_mse])
    axes[i].set_title(f"Complexity {c} - MSE")
    axes[i].grid()

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Time Plots
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, c in enumerate(complexities):
    _, _, lw_time, lstm_time = results[i]

    axes[i].bar(["Lightweight", "LSTM"], [lw_time, lstm_time])
    axes[i].set_title(f"Complexity {c} - Time")
    axes[i].grid()

plt.tight_layout()
plt.show()

# -----------------------------
# 8. Prediction Plots
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, c in enumerate(complexities):
    lw_p, lstm_p, y_true = all_preds[i]

    axes[i].plot(y_true[:100], label="Actual")
    axes[i].plot(lw_p[:100], label="Lightweight")
    axes[i].plot(lstm_p[:100], label="LSTM")

    axes[i].set_title(f"Complexity {c} - Prediction")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()