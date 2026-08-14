import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Generate Complex Data
# -----------------------------
np.random.seed(42)
t = np.arange(0, 300)

# Outdoor temp (nonlinear + seasonal)
outdoor = 10 + 8*np.sin(2*np.pi*t/50) + 3*np.sin(2*np.pi*t/15) + np.random.normal(0, 1, 300)

# Humidity (new feature)
humidity = 50 + 20*np.sin(2*np.pi*t/60) + np.random.normal(0, 5, 300)

# HVAC system (on/off behavior)
hvac = (outdoor > 18).astype(int)  # turns ON when hot

# Indoor temperature (nonlinear + memory + interaction)
indoor = []
for i in range(300):
    if i < 5:
        indoor.append(outdoor[i])
    else:
        val = (
            0.5 * indoor[-1] +
            0.2 * outdoor[i-2] +
            0.1 * outdoor[i-4] +
            0.05 * humidity[i] +
            -2 * hvac[i] +                      # cooling effect
            0.02 * (outdoor[i]**2) / 10 +      # nonlinear term
            np.random.normal(0, 0.8)
        )
        indoor.append(val)

indoor = np.array(indoor)

# Create dataframe
data = pd.DataFrame({
    "outdoor": outdoor,
    "humidity": humidity,
    "hvac": hvac,
    "indoor": indoor
})

# -----------------------------
# Step 2: Lag Features
# -----------------------------
for lag in range(1, 5):
    data[f"outdoor_lag{lag}"] = data["outdoor"].shift(lag)

data = data.dropna().reset_index(drop=True)

# Features
X = data.drop(columns=["indoor"])
y = data["indoor"]

# Train-test split
split = int(0.8 * len(data))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scaling (for NN)
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
# Step 4: Lightweight Model (Better NN)
# -----------------------------
start = time.time()
light_model = MLPRegressor(
    hidden_layer_sizes=(16, 8),   # slightly deeper
    activation='relu',
    max_iter=800,
    random_state=42
)
light_model.fit(X_train_scaled, y_train)
light_time = time.time() - start

light_pred = light_model.predict(X_test_scaled)
light_rmse = np.sqrt(mean_squared_error(y_test, light_pred))

# -----------------------------
# Step 5: Results
# -----------------------------
print("===== COMPLEX DATA COMPARISON =====")
print(f"Ridge RMSE: {ridge_rmse:.4f}")
print(f"Lightweight NN RMSE: {light_rmse:.4f}")
print()
print(f"Ridge Time: {ridge_time:.6f} sec")
print(f"Lightweight Time: {light_time:.6f} sec")

# -----------------------------
# Step 6: Plot
# -----------------------------
plt.figure()
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(ridge_pred, label="Ridge", linestyle='--')
plt.plot(light_pred, label="Lightweight NN", linestyle=':')
plt.legend()
plt.title("Complex Scenario Comparison")
plt.show()