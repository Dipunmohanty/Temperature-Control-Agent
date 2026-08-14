import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. Synthetic Data Generator
# (captures indoor/outdoor + time pattern)
# -----------------------------
def generate_series(n=400):
    t = np.arange(n)
    outside = 30 + 5*np.sin(2*np.pi*t/50)          # seasonal/periodic
    indoor = []

    temp = 25
    for i in range(n):
        # indoor follows outside with inertia
        temp = temp + 0.05*(outside[i] - temp) + np.random.normal(0, 0.2)
        indoor.append(temp)

    return np.array(indoor), outside

indoor, outside = generate_series()

# -----------------------------
# 2. Prepare Dataset (sequence → next value)
# -----------------------------
window = 5

X = []
y = []

for i in range(len(indoor) - window):
    seq = indoor[i:i+window]
    X.append(seq)
    y.append(indoor[i+window])

X = np.array(X)
y = np.array(y)

# Normalize (important for learning)
mean = X.mean()
std = X.std()

X = (X - mean) / std
y = (y - mean) / std

# -----------------------------
# 3. Lightweight Sequence Model
# (Linear AR model: like simplified LSTM)
# -----------------------------
w = np.random.randn(window)
b = 0.0

lr = 0.01
epochs = 50

losses = []

# -----------------------------
# 4. Training Loop
# -----------------------------
for epoch in range(epochs):

    total_loss = 0

    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        # prediction
        y_pred = np.dot(w, x_i) + b

        # loss (MSE)
        error = y_pred - y_i
        loss = error**2
        total_loss += loss

        # gradient
        dw = 2 * error * x_i
        db = 2 * error

        # update
        w -= lr * dw
        b -= lr * db

    losses.append(total_loss / len(X))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

# -----------------------------
# 5. Prediction
# -----------------------------
preds = []

for i in range(len(X)):
    y_pred = np.dot(w, X[i]) + b
    preds.append(y_pred)

preds = np.array(preds)

# denormalize
preds = preds * std + mean
y_true = y * std + mean

# -----------------------------
# 6. Plot: Prediction vs Actual
# -----------------------------
plt.figure()
plt.plot(y_true[:100], label="Actual")
plt.plot(preds[:100], label="Predicted")
plt.title("Paper 6 Style Temperature Prediction")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.grid()
plt.savefig("paper6_prediction.png")
plt.close()

# -----------------------------
# 7. Plot: Training Loss
# -----------------------------
plt.figure()
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.savefig("paper6_loss.png")
plt.close()

print("\nDone!")
print("Saved graphs:")
print(" - paper6_prediction.png")
print(" - paper6_loss.png")