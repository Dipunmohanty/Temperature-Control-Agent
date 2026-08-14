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
# 1. Sort time
df = df.sort_values('Time')

# 2. Create lag features (past)
df['temp_lag1'] = df['Temp'].shift(1)
df['temp_lag2'] = df['Temp'].shift(2)

# 3. Create future target
df['Temp_future'] = df['Temp'].shift(-1)

# 4. Drop NaNs (ONLY once at the end)
df = df.dropna()

# =========================
# 3. DEFINE FEATURES
# =========================
features = ['Temp', 'temp_lag1', 'temp_lag2', 'Humid', 'Pressure']

X = df[features]
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
print("\n MODEL PARAMETERS ")
print("Bias:", model.intercept_)
print("Weights:", model.coef_)

print("\nFeature order :")
for i, f in enumerate(features):
    print(f"x[{i}] -> {f} = {model.coef_[i]}")

# =========================
# 8. SAVE MODEL (OPTIONAL)
# =========================
import joblib
joblib.dump(model, "ridge_model_future_pred.pkl")

print("\nModel saved as ridge_model.pkl")

features = ['Temp', 'temp_lag1', 'temp_lag2', 'Humid', 'Pressure']
weights = model.coef_
bias = model.intercept_

with open("ridge_model.c", "w") as f:
    f.write("// Auto-generated Ridge model\n\n")
    
    f.write("float predict(float x[]) {\n")
    f.write(f"    float y = {bias:.6f}f;\n\n")

    for i, w in enumerate(weights):
        f.write(f"    y += {w:.6f}f * x[{i}]; // {features[i]}\n")

    f.write("\n    return y;\n")
    f.write("}\n")

print("C code generated: ridge_model.c")