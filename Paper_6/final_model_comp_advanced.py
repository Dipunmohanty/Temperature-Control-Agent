import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

np.random.seed(42)

# -----------------------------
# 1. Dataset Generator (IMPROVED)
# -----------------------------
def generate_data(n=600, complexity=0):
    t = np.arange(n)

    outside = 30 + 3*np.sin(2*np.pi*t/80)

    if complexity >= 1:
        outside += 2*np.sin(2*np.pi*t/20)

    if complexity >= 2:
        # sudden spikes
        spikes = np.zeros(n)
        idx = np.random.choice(n, size=15, replace=False)
        spikes[idx] = np.random.uniform(-6, 6, 15)
        outside += spikes

    if complexity >= 3:
        # rapid variation + strong noise
        outside += 4*np.sin(2*np.pi*t/3)
        outside += np.random.normal(0, 2.0, n)

    # indoor temp (thermal inertia + delay)
    indoor = []
    temp = 25

    for i in range(n):
        prev = indoor[-1] if i > 0 else temp
        temp = prev + 0.08*(outside[i] - prev) + 0.02*np.sin(2*np.pi*i/10)
        indoor.append(temp)

    return np.array(indoor), outside

# -----------------------------
# 2. Prepare Data (LONGER MEMORY)
# -----------------------------
def prepare_data(series, window=15):
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
# 4. LSTM Model (STRONGER)
# -----------------------------
def train_lstm(X, y):
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()

    Xn = (X - X_mean) / X_std
    yn = (y - y_mean) / y_std

    Xn = Xn.reshape((Xn.shape[0], Xn.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xn.shape[1], 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    start = time.time()

    model.fit(Xn, yn, epochs=50, verbose=0)

    duration = time.time() - start

    preds = model.predict(Xn, verbose=0).flatten()
    preds = preds * y_std + y_mean

    return preds, duration

# -----------------------------
# 5. Run Experiment
# -----------------------------
complexities = [0, 1, 2, 3]

results = []
all_preds = []

# -----------------------------
# Dataset Visualization
# -----------------------------
plt.figure(figsize=(10,6))
for c in complexities:
    indoor, outside = generate_data(complexity=c)
    plt.plot(outside[:200], label=f"Complexity {c}")
plt.title("Dataset (Outside Temperature)")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Training Loop
# -----------------------------
for c in complexities:
    print(f"\n--- Complexity {c} ---")

    indoor, outside = generate_data(complexity=c)
    X, y = prepare_data(indoor)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lw_preds_train, lw_time = train_lightweight(X_train, y_train)
    lstm_preds_train, lstm_time = train_lstm(X_train, y_train)

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
fig, axes = plt.subplots(2, 2, figsize=(12,8))
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
fig, axes = plt.subplots(2, 2, figsize=(12,8))
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
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes = axes.flatten()

for i, c in enumerate(complexities):
    lw_p, lstm_p, y_true = all_preds[i]

    axes[i].plot(y_true[:120], label="Actual")
    axes[i].plot(lw_p[:120], label="Lightweight")
    axes[i].plot(lstm_p[:120], label="LSTM")

    axes[i].set_title(f"Complexity {c} - Prediction")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()