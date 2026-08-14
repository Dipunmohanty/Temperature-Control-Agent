import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("GARG_EMY.csv")

df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time')

# =========================
# 2. FEATURE ENGINEERING
# =========================
# Time features
df['hour'] = df['Time'].dt.hour
df['day'] = df['Time'].dt.dayofweek

# Lag features (critical)
for i in range(1, 4):
    df[f'temp_lag{i}'] = df['Temp'].shift(i)

df['humid_lag1'] = df['Humid'].shift(1)
df['pressure_lag1'] = df['Pressure'].shift(1)

df = df.dropna()

# =========================
# 3. SPLIT DATA (TIME SERIES)
# =========================
target = 'Temp'

X = df.drop(['Time', target], axis=1)
y = df[target]

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 4. METRICS
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# 5. RIDGE MODEL
# =========================
start = time.time()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_time = time.time() - start

# =========================
# 6. LIGHTWEIGHT MODEL (LINEAR)
# =========================
start = time.time()

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

linear_time = time.time() - start

# =========================
# 7. LSTM MODEL (PROPER SEQUENCE)
# =========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

SEQ_LEN = 5

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X.iloc[i:i+seq_len].values)
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LEN)

start = time.time()

lstm = Sequential([
    LSTM(32, activation='relu', input_shape=(SEQ_LEN, X.shape[1])),
    Dense(1)
])

lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_seq, y_train_seq, epochs=10, verbose=0)

lstm_pred = lstm.predict(X_test_seq)

lstm_time = time.time() - start

# =========================
# 8. RESULTS
# =========================
print("\n===== MODEL COMPARISON =====")

print("\nRidge:")
print("RMSE:", rmse(y_test, ridge_pred))
print("Time:", ridge_time)

print("\nLinear (Lightweight):")
print("RMSE:", rmse(y_test, linear_pred))
print("Time:", linear_time)

print("\nLSTM:")
print("RMSE:", rmse(y_test_seq, lstm_pred))
print("Time:", lstm_time)

# =========================
# 9. PLOT PREDICTIONS
# =========================
plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual")

plt.plot(ridge_pred, label="Ridge")
plt.plot(linear_pred, label="Linear")

# LSTM aligned (shift due to sequence)
plt.plot(range(SEQ_LEN, len(y_test)), lstm_pred, label="LSTM")

plt.legend()
plt.title("Model Prediction Comparison")
plt.xlabel("Time Steps")
plt.ylabel("Temperature")
plt.show()