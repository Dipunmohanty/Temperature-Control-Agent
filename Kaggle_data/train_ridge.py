import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# =========================
# 1. LOAD DATA (MULTI FILE)
# =========================
files = [
    "GARG_EMY.csv", "KZNA_EMY.csv", "OFFC_EMY.csv",
    "YARD_EMY.csv", "YPN_EMY.csv", "YPN2_EMY.csv",
    "YPN3_EMY.csv", "bath_EMY.csv"
]

dfs = []

for f in files:
    df = pd.read_csv(f)
    df['Time'] = pd.to_datetime(df['Time'])

    # keep only relevant sensor data
    df = df[['Time', 'Temp', 'Humid', 'Pressure']]
    dfs.append(df)

# combine datasets
df = pd.concat(dfs, ignore_index=True)

# =========================
# 2. SORT + FEATURE ENGINEERING
# =========================
df = df.sort_values('Time')

# create lag features (important!)
df['temp_lag1'] = df['Temp'].shift(1)
df['temp_lag2'] = df['Temp'].shift(2)

# remove first rows with NaN
df = df.dropna()

print("Data shape:", df.shape)

# =========================
# 3. DEFINE FEATURES
# =========================
features = ['Temp', 'temp_lag1', 'temp_lag2', 'Humid', 'Pressure']

X = df[features]
# y = df['Temp']
df['Temp_future'] = df['Temp'].shift(-1)
df = df.dropna()

y = df['Temp_future']

# =========================
# 4. TRAIN-TEST SPLIT (TIME BASED)
# =========================
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# 5. TRAIN MODEL
# =========================
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# =========================
# 6. PREDICT + EVALUATE
# =========================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", rmse)

# =========================
# 7. PRINT MODEL (FOR DEPLOYMENT)
# =========================
print("\n=== MODEL PARAMETERS ===")
print("Bias:", model.intercept_)
print("Weights:", model.coef_)

print("\nFeature order (IMPORTANT):")
for i, f in enumerate(features):
    print(f"x[{i}] -> {f}")

# =========================
# 8. SAVE MODEL (OPTIONAL)
# =========================
import joblib
joblib.dump(model, "ridge_model.pkl")

print("\nModel saved as ridge_model.pkl")