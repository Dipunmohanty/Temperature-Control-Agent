import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X.iloc[i:i+seq_len].values.astype(np.float32))
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)

# =========================
# 1. LOAD DATASETS
# =========================
# files = [
#     "GARG_EMY.csv",
#     "KZNA_EMY.csv",
#     "OFFC_EMY.csv",
#     "YARD_EMY.csv",
#     "YPN_EMY.csv",
#     "YPN2_EMY.csv",
#     "YPN3_EMY.csv",
#     "bath_EMY.csv"
# ]

# dfs = []

# for f in files:
#     df_temp = pd.read_csv(f)
#     df_temp['Time'] = pd.to_datetime(df_temp['Time'])
#     df_temp['source'] = f
#     dfs.append(df_temp)

# df = pd.concat(dfs, ignore_index=True)


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
    df_temp = pd.read_csv(f)

    df_temp['Time'] = pd.to_datetime(df_temp['Time'])

    # Keep only common columns
    df_temp = df_temp[['Time', 'Temp', 'Humid', 'Pressure']]

    # Add source label
    df_temp['source'] = f

    dfs.append(df_temp)

# STACK (not merge)
df = pd.concat(dfs, ignore_index=True)

print("Columns after fix:", df.columns)
print("Shape:", df.shape)


# =========================
# 2. SORT PROPERLY
# =========================
# df = df.sort_values(['source', 'Time'])

# =========================
# 3. FEATURE ENGINEERING (FIXED)
# =========================
# df['hour'] = df['Time'].dt.hour
# df['day'] = df['Time'].dt.dayofweek

# # Lag features per dataset
# for i in range(1, 4):
#     df[f'temp_lag{i}'] = df.groupby('source')['Temp'].shift(i)

# df['humid_lag1'] = df.groupby('source')['Humid'].shift(1)
# df['pressure_lag1'] = df.groupby('source')['Pressure'].shift(1)

# # ONLY drop missing target
# df = df.dropna(subset=['Temp'])

# # Fill missing values safely (per dataset)
# # df = df.groupby('source').apply(lambda x: x.fillna(method='bfill').fillna(method='ffill'))
# # df = df.reset_index(drop=True)
# # Fill missing values safely per dataset
# # df = df.groupby('source').apply(lambda x: x.bfill().ffill())

# # # Fix index after groupby apply
# # df = df.reset_index(drop=True)

# df = df.groupby('source', group_keys=False).apply(lambda x: x.bfill().ffill())

# print("After cleaning:", df.shape)

# =========================
# SAFE CLEANING (NO GROUPBY APPLY)
# =========================

# Sort properly
df = df.sort_values(['source', 'Time'])

# Lag features per dataset
for i in range(1, 4):
    df[f'temp_lag{i}'] = df.groupby('source')['Temp'].shift(i)

df['humid_lag1'] = df.groupby('source')['Humid'].shift(1)
df['pressure_lag1'] = df.groupby('source')['Pressure'].shift(1)

# Drop only rows where lagging created NaNs (first few rows per dataset)
df = df.dropna()

print("After cleaning:", df.shape)
print(df.columns)

# =========================
# 4. SPLIT DATA
# =========================
# target = 'Temp'

# X = df.drop(['Time', target], axis=1)
# print(df.columns)
# y = df[target]

# # Encode source
# X = pd.get_dummies(X, columns=['source'])

# split = int(len(df) * 0.8)

# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# =========================
# PREPARE DATA (SAFE FOR LSTM)
# =========================

X = df.drop(['Time', 'Temp'], axis=1)
y = df['Temp']

# Encode source
X = pd.get_dummies(X, columns=['source'])

# Convert EVERYTHING to numeric
X = X.astype(np.float32)
y = y.astype(np.float32)

print("Data types:\n", X.dtypes)

# =========================
# TRAIN-TEST SPLIT (TIME BASED)
# =========================
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# 5. METRIC
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# 6. RIDGE
# =========================
start = time.time()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_time = time.time() - start

# =========================
# 7. LIGHTWEIGHT (LINEAR)
# =========================
start = time.time()

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

linear_time = time.time() - start

# =========================
# 8. LSTM
# =========================
# =========================
# LSTM (IMPROVED VERSION)
# =========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. SCALE DATA (CRITICAL)
# -------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 2. BIG SEQUENCE LENGTH
# -------------------------
SEQ_LEN = 24   # 24 hours (daily pattern)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

# Convert to numpy (important after scaling)
X_train_np = np.array(X_train_scaled)
X_test_np = np.array(X_test_scaled)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

X_train_seq, y_train_seq = create_sequences(X_train_np, y_train_np, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test_np, y_test_np, SEQ_LEN)

# -------------------------
# 3. BETTER LSTM MODEL
# -------------------------
import time
start = time.time()

lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_train_seq.shape[2])),
    Dropout(0.2),
    
    LSTM(32),
    Dense(1)
])

lstm.compile(optimizer='adam', loss='mse')

lstm.fit(
    X_train_seq, y_train_seq,
    epochs=20,
    batch_size=64,
    verbose=1
)

lstm_pred = lstm.predict(X_test_seq)

lstm_time = time.time() - start

# =========================
# 9. RESULTS
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

print("\n===== MULTI-DATASET RESULTS =====")
print(results)

results.to_csv("multi_dataset_results.csv", index=False)

# =========================
# 10. PLOTS
# =========================
plt.figure()
plt.bar(results["Model"], results["RMSE"])
plt.title("RMSE Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.show()

plt.figure()
plt.bar(results["Model"], results["Runtime (s)"])
plt.title("Runtime Comparison")
plt.xlabel("Model")
plt.ylabel("Time (s)")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual")
plt.plot(ridge_pred, label="Ridge")
plt.plot(linear_pred, label="Linear")

offset = len(y_test) - len(lstm_pred)
plt.plot(range(offset, offset + len(lstm_pred)), lstm_pred, label="LSTM")

plt.legend()
plt.title("Prediction Comparison")
plt.show()