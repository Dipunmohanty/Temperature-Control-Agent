import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import time

np.random.seed(42)

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    duration = time.time() - start
    return result, duration

# -----------------------------
# 1. Dataset Generator
# -----------------------------
def generate_data(n=600, complexity=0):
    t = np.arange(n)

    outside = 30 + 3*np.sin(2*np.pi*t/80)

    if complexity >= 1:
        outside += 2*np.sin(2*np.pi*t/20)

    if complexity >= 2:
        spikes = np.zeros(n)
        idx = np.random.choice(n, size=15, replace=False)
        spikes[idx] = np.random.uniform(-6, 6, 15)
        outside += spikes

    if complexity >= 3:
        outside += 4*np.sin(2*np.pi*t/3)
        outside += np.random.normal(0, 2.0, n)

    indoor = []
    temp = 25

    for i in range(n):
        prev = indoor[-1] if i > 0 else temp
        temp = prev + 0.08*(outside[i] - prev)
        indoor.append(temp)

    return np.array(indoor), outside


# -----------------------------
# 2. Feature Engineering
# -----------------------------
def prepare_features(indoor, outside, window=15):
    X, y = [], []

    for i in range(len(indoor) - window):
        temp_seq = indoor[i:i+window]
        out_seq = outside[i:i+window]

        trend = temp_seq[-1] - temp_seq[0]
        mean_temp = np.mean(temp_seq)
        std_temp = np.std(temp_seq)

        features = np.concatenate([
            temp_seq,
            out_seq,
            [trend, mean_temp, std_temp]
        ])

        X.append(features)
        y.append(indoor[i+window])

    return np.array(X), np.array(y)


# -----------------------------
# 3. Lightweight Model
# -----------------------------
# def train_lightweight(X, y):
#     Xn = (X - X.mean()) / X.std()
#     yn = (y - y.mean()) / y.std()

#     w = np.random.randn(X.shape[1])
#     b = 0
#     lr = 0.005

#     start = time.time()

#     for _ in range(30):
#         for i in range(len(Xn)):
#             pred = np.dot(w, Xn[i]) + b
#             err = pred - yn[i]
#             w -= lr * 2 * err * Xn[i]
#             b -= lr * 2 * err

#     duration = time.time() - start

#     preds = np.dot(Xn, w) + b
#     preds = preds * y.std() + y.mean()

#     return preds, duration

# def train_lightweight(X_train, y_train, X_test):
#     import time
#     start = time.time()

#     X_mean, X_std = X_train.mean(), X_train.std()
#     y_mean, y_std = y_train.mean(), y_train.std()

#     Xn = (X_train - X_mean) / X_std
#     yn = (y_train - y_mean) / y_std

#     w = np.random.randn(X_train.shape[1])
#     b = 0
#     lr = 0.005

#     losses = []  # 👈 track loss

#     for _ in range(30):
#         epoch_loss = 0

#         for i in range(len(Xn)):
#             pred = np.dot(w, Xn[i]) + b
#             err = pred - yn[i]

#             epoch_loss += err**2

#             w -= lr * 2 * err * Xn[i]
#             b -= lr * 2 * err

#         losses.append(epoch_loss / len(Xn))

#     # Predict
#     X_test_n = (X_test - X_mean) / X_std
#     preds = np.dot(X_test_n, w) + b
#     preds = preds * y_std + y_mean

#     duration = time.time() - start


#     return preds, losses, duration

def train_lightweight(X_train, y_train, X_test):
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    Xn = (X_train - X_mean) / X_std
    yn = (y_train - y_mean) / y_std

    w = np.random.randn(X_train.shape[1])
    b = 0
    lr = 0.005

    losses = []

    for _ in range(30):
        epoch_loss = 0
        for i in range(len(Xn)):
            pred = np.dot(w, Xn[i]) + b
            err = pred - yn[i]
            epoch_loss += err**2

            w -= lr * 2 * err * Xn[i]
            b -= lr * 2 * err

        losses.append(epoch_loss / len(Xn))

    # Predict
    X_test_n = (X_test - X_mean) / X_std
    preds = np.dot(X_test_n, w) + b
    preds = preds * y_std + y_mean

    return preds, losses


# -----------------------------
# 4. Sklearn Models
# -----------------------------
def train_model(model, X_train, y_train, X_test):
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    duration = time.time() - start
    return preds


# -----------------------------
# 5. LSTM Model
# -----------------------------
# def train_lstm(X, y):
#     X_mean, X_std = X.mean(), X.std()
#     y_mean, y_std = y.mean(), y.std()

#     Xn = (X - X_mean) / X_std
#     yn = (y - y_mean) / y_std

#     Xn = Xn.reshape((Xn.shape[0], Xn.shape[1], 1))

#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(Xn.shape[1], 1)),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(1)
#     ])

#     model.compile(optimizer='adam', loss='mse')

#     start = time.time()

#     model.fit(Xn, yn, epochs=40, verbose=0)

#     duration = time.time() - start

#     preds = model.predict(Xn, verbose=0).flatten()
#     preds = preds * y_std + y_mean

#     return preds, duration

# def train_lstm(X_train, y_train, X_test):
#     X_mean, X_std = X_train.mean(), X_train.std()
#     y_mean, y_std = y_train.mean(), y_train.std()

#     Xn = (X_train - X_mean) / X_std
#     yn = (y_train - y_mean) / y_std

#     Xn = Xn.reshape((Xn.shape[0], Xn.shape[1], 1))

#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(Xn.shape[1], 1)),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(1)
#     ])

#     model.compile(optimizer='adam', loss='mse')

#     history = model.fit(Xn, yn, epochs=20, verbose=0)

#     # Predict
#     X_test_n = (X_test - X_mean) / X_std
#     X_test_n = X_test_n.reshape((X_test_n.shape[0], X_test_n.shape[1], 1))

#     preds = model.predict(X_test_n, verbose=0).flatten()
#     preds = preds * y_std + y_mean

#     return preds, history.history['loss']

# def train_lstm(X_train, y_train, X_test):
#     import time
#     start = time.time()

#     X_mean, X_std = X_train.mean(), X_train.std()
#     y_mean, y_std = y_train.mean(), y_train.std()

#     Xn = (X_train - X_mean) / X_std
#     yn = (y_train - y_mean) / y_std

#     Xn = Xn.reshape((Xn.shape[0], Xn.shape[1], 1))

#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(Xn.shape[1], 1)),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(1)
#     ])

#     model.compile(optimizer='adam', loss='mse')

#     history = model.fit(Xn, yn, epochs=20, verbose=0)

#     duration = time.time() - start

#     # Predict on TEST
#     X_test_n = (X_test - X_mean) / X_std
#     X_test_n = X_test_n.reshape((X_test_n.shape[0], X_test_n.shape[1], 1))

#     preds = model.predict(X_test_n, verbose=0).flatten()
#     preds = preds * y_std + y_mean

#     losses = history.history['loss']

#     return preds, losses, duration

def train_lstm(X_train, y_train, X_test):
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    Xn = (X_train - X_mean) / X_std
    yn = (y_train - y_mean) / y_std

    Xn = Xn.reshape((Xn.shape[0], Xn.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xn.shape[1], 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(Xn, yn, epochs=20, verbose=0)

    # Predict
    X_test_n = (X_test - X_mean) / X_std
    X_test_n = X_test_n.reshape((X_test_n.shape[0], X_test_n.shape[1], 1))

    preds = model.predict(X_test_n, verbose=0).flatten()
    preds = preds * y_std + y_mean

    return preds, history.history['loss']


# -----------------------------
# 6. Experiment
# -----------------------------
complexities = [0, 1, 2, 3]

for c in complexities:
    print(f"\n--- Complexity {c} ---")

    indoor, outside = generate_data(complexity=c)
    X, y = prepare_features(indoor, outside)

    # -----------------------------
    # Plot Dataset
    # ----------------------------- 
    plt.figure(figsize=(10,4))
    plt.plot(indoor[:500], label="Indoor Temp")
    plt.plot(outside[:500], label="Outside Temp")
    plt.title(f"Complexity {c} - Dataset (First 500 points)")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid()
    plt.show()

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Lightweight
    # lw_preds_train, lw_time = train_lightweight(X_train, y_train)\
    # lw_preds, lw_losses = train_lightweight(X_train, y_train, X_test)
    # lw_preds = lw_preds_train[-len(y_test):]
    # lw_preds, lw_losses, lw_time = train_lightweight(X_train, y_train, X_test)

    # Lightweight
    # lw_preds, lw_losses, lw_time = train_lightweight(X_train, y_train, X_test)

    # # Ridge
    # ridge_preds, ridge_time = train_model(Ridge(alpha=1.0), X_train, y_train, X_test)

    # # Decision Tree
    # tree_preds, tree_time = train_model(
    #     DecisionTreeRegressor(max_depth=5), X_train, y_train, X_test)

    # # MLP
    # mlp_preds, mlp_time = train_model(
    #     MLPRegressor(hidden_layer_sizes=(32,), max_iter=300),
    #     X_train, y_train, X_test)

    # # LSTM
    # lstm_preds, lstm_time = train_lstm(X_train, y_train, X_test)
    # lstm_preds = lstm_preds[-len(y_test):]
    # lstm_preds, lstm_losses = train_lstm(X_train, y_train, X_test)

    # # Lightweight
    # lw_preds, lw_losses, lw_time = train_lightweight(X_train, y_train, X_test)

    # # Ridge
    # ridge_preds, ridge_time = train_model(Ridge(alpha=1.0), X_train, y_train, X_test)

    # # Tree
    # tree_preds, tree_time = train_model(
    # DecisionTreeRegressor(max_depth=5), X_train, y_train, X_test)

    #  MLP
    # mlp_preds, mlp_time = train_model(
    # MLPRegressor(hidden_layer_sizes=(32,), max_iter=300),
    # X_train, y_train, X_test)

    # # LSTM (FIXED)
    # lstm_preds, lstm_losses, lstm_time = train_lstm(X_train, y_train, X_test)

    # Lightweight
    (lw_preds, lw_losses), lw_time = measure_time(
    train_lightweight, X_train, y_train, X_test)

# Ridge
    ridge_preds, ridge_time = measure_time(
    train_model, Ridge(alpha=1.0), X_train, y_train, X_test)

# Tree
    tree_preds, tree_time = measure_time(
    train_model, DecisionTreeRegressor(max_depth=5),
    X_train, y_train, X_test
)

# MLP
    mlp_preds, mlp_time = measure_time(
    train_model,
    MLPRegressor(hidden_layer_sizes=(32,), max_iter=300),
    X_train, y_train, X_test
)

# LSTM
    (lstm_preds, lstm_losses), lstm_time = measure_time(
        train_lstm, X_train, y_train, X_test
)

    # -----------------------------
    # Compute MSE
    # -----------------------------
    def mse(a, b): return np.mean((a - b)**2)

    # results = {
    #     "Lightweight": (mse(lw_preds, y_test), lw_time),
    #     "Ridge": (mse(ridge_preds, y_test), ridge_time),
    #     "Tree": (mse(tree_preds, y_test), tree_time),
    #     "MLP": (mse(mlp_preds, y_test), mlp_time),
    #     "LSTM": (mse(lstm_preds, y_test), lstm_time)
    # }

    results = {
    "Lightweight": (mse(lw_preds, y_test), lw_time),
    "Ridge": (mse(ridge_preds, y_test), ridge_time),
    "Tree": (mse(tree_preds, y_test), tree_time),
    "MLP": (mse(mlp_preds, y_test), mlp_time),
    "LSTM": (mse(lstm_preds, y_test), lstm_time)
    }

    # -----------------------------
    # Print Results
    # -----------------------------
    # for k, v in results.items():
    #     print(f"{k:12} → MSE: {v[0]:.3f}, Time: {v[1]:.2f}s")

    # for k, v in results.items():
    #     print(f"{k:12} → MSE: {v[0]:.3f}, Time: {v[1]:.2f}s")

    for k, v in results.items():
        print(f"{k:12} → MSE: {v[0]:.3f}, Time: {v[1]:.2f}s")
    
    # Plot losses for lightweight training
    plt.figure()
    plt.plot(lw_losses)
    plt.title(f"Complexity {c} - Lightweight Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(lstm_losses)
    plt.title(f"Complexity {c} - LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(lw_losses, label="Lightweight Loss")
    plt.plot(lstm_losses, label="LSTM Loss")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # Plot MSE
    # -----------------------------
    plt.figure()
    names = list(results.keys())
    values = [v[0] for v in results.values()]
    plt.bar(names, values)
    plt.title(f"Complexity {c} - MSE Comparison")
    plt.xticks(rotation=30)
    plt.grid()
    plt.show()

    plt.figure()
    names = list(results.keys())
    times = [v[1] for v in results.values()]

    plt.bar(names, times)
    plt.title("Model Time Comparison")
    plt.ylabel("Seconds")
    plt.xticks(rotation=30)
    plt.grid() 
    plt.show()

    # -----------------------------
    # Plot Predictions
    # -----------------------------
    plt.figure(figsize=(10,5))
    plt.plot(y_test[:120], label="Actual")
    plt.plot(lw_preds[:120], label="Lightweight")
    plt.plot(ridge_preds[:120], label="Ridge")
    plt.plot(tree_preds[:120], label="Tree")
    plt.plot(mlp_preds[:120], label="MLP")
    plt.plot(lstm_preds[:120], label="LSTM")

    plt.title(f"Complexity {c} - Predictions")
    plt.legend()
    plt.grid()
    plt.show()