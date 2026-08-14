import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Generate Data
# -----------------------------
np.random.seed(42)

time_arr = np.arange(0, 200)

outdoor_temp = 10 + 10 * np.sin(2 * np.pi * time_arr / 50) + np.random.normal(0, 1, 200)

indoor_temp = []
for t in range(200):
    if t < 3:
        indoor_temp.append(outdoor_temp[t])
    else:
        val = 0.7 * indoor_temp[-1] + 0.3 * outdoor_temp[t-2] + np.random.normal(0, 0.5)
        indoor_temp.append(val)

indoor_temp = np.array(indoor_temp)

data = pd.DataFrame({
    "outdoor": outdoor_temp,
    "indoor": indoor_temp
})

# -----------------------------
# Step 2: Add Lag Features
# -----------------------------
data["lag1"] = data["outdoor"].shift(1)
data["lag2"] = data["outdoor"].shift(2)

data = data.dropna().reset_index(drop=True)

X = data[["outdoor", "lag1", "lag2"]]
y = data["indoor"]

# Train-test split
split = int(0.8 * len(data))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 3: Ridge Model
# -----------------------------
start = time.time()
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_time = time.time() - start

ridge_pred = ridge.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

# -----------------------------
# Step 4: Lightweight Model (Tiny NN)
# -----------------------------
start = time.time()
light_model = MLPRegressor(hidden_layer_sizes=(8,),  # very small model
                           activation='relu',
                           max_iter=500,
                           random_state=42)

light_model.fit(X_train_scaled, y_train)
light_time = time.time() - start

light_pred = light_model.predict(X_test_scaled)
light_rmse = np.sqrt(mean_squared_error(y_test, light_pred))

# -----------------------------
# Step 5: Results
# -----------------------------
print("===== MODEL COMPARISON =====")
print(f"Ridge RMSE: {ridge_rmse:.4f}")
print(f"Lightweight RMSE: {light_rmse:.4f}")
print()
print(f"Ridge Training Time: {ridge_time:.6f} sec")
print(f"Lightweight Training Time: {light_time:.6f} sec")

# -----------------------------
# Step 6: Plot Comparison
# -----------------------------
plt.figure()
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(ridge_pred, label="Ridge", linestyle='--')
plt.plot(light_pred, label="Lightweight NN", linestyle=':')
plt.legend()
plt.title("Model Comparison (Prediction)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()