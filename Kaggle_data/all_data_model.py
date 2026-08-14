import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

# =========================
# 1. LOAD MULTIPLE FILES
# =========================
files = [
    "GARG_EMY.csv",
    "KZNA_EMY.csv",
    "OFFC_EMY.csv",
    "YARD_EMY.csv",
    "YPN_EMY.csv",
    "YPN2_EMY.csv",
    "YPN3_EMY.csv",
    "bath_EMY.csv"
]

dfs = []

for f in files:
    df = pd.read_csv(f)
    df['Time'] = pd.to_datetime(df['Time'])
    df['source'] = f   # track dataset source
    dfs.append(df)

# Combine all datasets
df = pd.concat(dfs, ignore_index=True)

# Sort by time (important)
df = df.sort_values('Time')

# =========================
# 2. FEATURE ENGINEERING
# =========================
df['hour'] = df['Time'].dt.hour
df['day'] = df['Time'].dt.dayofweek

# Lag features (grouped by source to avoid mixing signals)
df = df.sort_values(['source', 'Time'])

for i in range(1, 4):
    df[f'temp_lag{i}'] = df.groupby('source')['Temp'].shift(i)

df['humid_lag1'] = df.groupby('source')['Humid'].shift(1)
df['pressure_lag1'] = df.groupby('source')['Pressure'].shift(1)

print("Before dropna:", df.shape)
df = df.dropna()
print("After dropna:", df.shape)

# =========================
# 3. SPLIT DATA
# =========================
target = 'Temp'

X = df.drop(['Time', target], axis=1)
y = df[target]

# Convert source to numeric
X = pd.get_dummies(X, columns=['source'])

# Time-based split
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 4. METRIC
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# 5. RIDGE
# =========================
start = time.time()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_time = time.time() - start

# =========================
# 6. LIGHTWEIGHT (LINEAR)
# =========================
start = time.time()

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

linear_time = time.time() - start

# =========================
# 7. LSTM (SEQUENCE)
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
results = pd.DataFrame({
    "Model": ["Ridge", "Linear", "LSTM"],
    "RMSE": [
        rmse(y_test, ridge_pred),
        rmse(y_test, linear_pred),
        rmse(y_test_seq, lstm_pred)
    ],
    "Runtime (s)": [ridge_time, linear_time, lstm_time]
})

print("\n===== MULTI-DATASET COMPARISON =====")
print(results)

# Save
results.to_csv("multi_dataset_results.csv", index=False)

# =========================
# 9. PLOTS
# =========================

# RMSE
plt.figure()
plt.bar(results["Model"], results["RMSE"])
plt.title("RMSE Comparison (Multi Dataset)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.show()

# Runtime
plt.figure()
plt.bar(results["Model"], results["Runtime (s)"])
plt.title("Runtime Comparison (Multi Dataset)")
plt.xlabel("Model")
plt.ylabel("Time (s)")
plt.show()

# Predictions
plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual")
plt.plot(ridge_pred, label="Ridge")
plt.plot(linear_pred, label="Linear")

offset = len(y_test) - len(lstm_pred)
plt.plot(range(offset, offset + len(lstm_pred)), lstm_pred, label="LSTM")

plt.legend()
plt.title("Prediction Comparison (Multi Dataset)")
plt.xlabel("Time Steps")
plt.ylabel("Temperature")
plt.show()