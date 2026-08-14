import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil, os

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# RAM FUNCTION
# =========================
def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

# =========================
# LOAD DATASETS (LONG FORMAT)
# =========================
files = [
    "GARG_EMY.csv", "KZNA_EMY.csv", "OFFC_EMY.csv",
    "YARD_EMY.csv", "YPN_EMY.csv", "YPN2_EMY.csv",
    "YPN3_EMY.csv", "bath_EMY.csv"
]

dfs = []
for f in files:
    temp = pd.read_csv(f)
    temp['Time'] = pd.to_datetime(temp['Time'])
    temp = temp[['Time','Temp','Humid','Pressure']]
    temp['source'] = f
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# =========================
# FEATURE ENGINEERING
# =========================
df = df.sort_values(['source','Time'])

df['hour'] = df['Time'].dt.hour
df['day'] = df['Time'].dt.dayofweek

for i in range(1,4):
    df[f'temp_lag{i}'] = df.groupby('source')['Temp'].shift(i)

df['humid_lag1'] = df.groupby('source')['Humid'].shift(1)
df['pressure_lag1'] = df.groupby('source')['Pressure'].shift(1)

df = df.dropna()

print("Data shape:", df.shape)

# =========================
# PREPARE DATA
# =========================
X = df.drop(['Time','Temp'], axis=1)
y = df['Temp']

X = pd.get_dummies(X, columns=['source'])

X = X.astype(np.float32)
y = y.astype(np.float32)

# =========================
# SPLIT
# =========================
split = int(len(X)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# METRIC
# =========================
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

# =========================
# RIDGE
# =========================
ram0 = get_ram()
start = time.time()

ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)

ridge_time = time.time()-start
ridge_ram = get_ram()-ram0
ridge_mem = ridge.coef_.nbytes/(1024*1024)

# =========================
# LINEAR
# =========================
ram0 = get_ram()
start = time.time()

linear = LinearRegression()
linear.fit(X_train,y_train)
linear_pred = linear.predict(X_test)

linear_time = time.time()-start
linear_ram = get_ram()-ram0
linear_mem = linear.coef_.nbytes/(1024*1024)

# =========================
# LSTM (IMPROVED)
# =========================
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

SEQ = 24

def create_seq(X,y,seq):
    Xs,ys=[],[]
    for i in range(len(X)-seq):
        Xs.append(X[i:i+seq])
        ys.append(y[i+seq])
    return np.array(Xs),np.array(ys)

X_train_seq,y_train_seq = create_seq(X_train_s,y_train.values,SEQ)
X_test_seq,y_test_seq = create_seq(X_test_s,y_test.values,SEQ)

ram0 = get_ram()
start = time.time()

lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ,X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

lstm.compile(optimizer='adam',loss='mse')
lstm.fit(X_train_seq,y_train_seq,epochs=20,batch_size=64,verbose=0)

lstm_pred = lstm.predict(X_test_seq)

lstm_time = time.time()-start
lstm_ram = get_ram()-ram0
lstm_mem = lstm.count_params()*4/(1024*1024)

# =========================
# RESULTS
# =========================
results = pd.DataFrame({
    "Model":["Ridge","Linear","LSTM"],
    "RMSE":[
        rmse(y_test,ridge_pred),
        rmse(y_test,linear_pred),
        rmse(y_test_seq,lstm_pred)
    ],
    "Runtime (s)":[ridge_time,linear_time,lstm_time],
    "RAM (MB)":[ridge_ram,linear_ram,lstm_ram],
    "Model Size (MB)":[ridge_mem,linear_mem,lstm_mem]
})

print("\n===== FINAL RESULTS =====")
print(results)

results.to_csv("final_results.csv",index=False)

# =========================
# PLOTS
# =========================
plt.figure()
plt.bar(results["Model"],results["RMSE"])
plt.title("RMSE Comparison")
plt.show()

plt.figure()
plt.bar(results["Model"],results["Runtime (s)"])
plt.title("Runtime Comparison")
plt.show()

plt.figure()
plt.bar(results["Model"],results["RAM (MB)"])
plt.title("RAM Usage")
plt.show()

plt.figure()
plt.bar(results["Model"],results["Model Size (MB)"])
plt.title("Model Size")
plt.show()

# Prediction plot
plt.figure(figsize=(12,6))
plt.plot(y_test.values,label="Actual")
plt.plot(ridge_pred,label="Ridge")
plt.plot(linear_pred,label="Linear")

offset=len(y_test)-len(lstm_pred)
plt.plot(range(offset,offset+len(lstm_pred)),lstm_pred,label="LSTM")

plt.legend()
plt.title("Prediction Comparison")
plt.show()