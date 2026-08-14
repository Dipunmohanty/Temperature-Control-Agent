import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1. Lightweight "LSTM-like" Predictor
# -----------------------------
def predict_temp(history):
    # weighted average (recent values more important)
    weights = np.linspace(1, 2, len(history))
    weights /= weights.sum()
    return np.dot(history, weights)

# -----------------------------
# 2. Environment Simulation
# -----------------------------
def environment_step(temp, action, outside_temp=30):
    # simulate thermal dynamics
    next_temp = temp + 0.1 * (outside_temp - temp) + action * 0.5
    energy = abs(action)
    return next_temp, energy

# -----------------------------
# 3. Discretization (State Space)
# -----------------------------
bins = np.linspace(15, 40, 25)

def get_state(temp):
    return np.digitize(temp, bins)

# Q-table: states × actions
Q = np.zeros((len(bins)+1, 3))

# -----------------------------
# 4. RL Parameters
# -----------------------------
alpha = 0.1
gamma = 0.9
epsilon = 0.2
target_temp = 25

rewards = []

print("Training Paper-2 style model (no TensorFlow)...")

# -----------------------------
# 5. Training Loop
# -----------------------------
for episode in range(200):

    temp = np.random.uniform(20, 35)
    history = [temp] * 5  # initial history
    total_reward = 0

    for step in range(30):

        # ---- Prediction step (Paper 2 idea) ----
        temp_pred = predict_temp(history)

        state = get_state(temp_pred)

        # ---- Action selection ----
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = np.argmax(Q[state])

        action = [-1, 0, 1][action_idx]

        # ---- Environment step ----
        next_temp, energy = environment_step(temp, action)

        # ---- Reward ----
        reward = -(abs(next_temp - target_temp) + 0.1 * energy)
        total_reward += reward

        next_state = get_state(next_temp)

        # ---- Q-learning update ----
        Q[state, action_idx] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action_idx]
        )

        # update history
        history.pop(0)
        history.append(next_temp)

        temp = next_temp

    rewards.append(total_reward)

# -----------------------------
# 6. Plot Learning Curve
# -----------------------------
plt.figure()
plt.plot(rewards)
plt.title("RL Learning Curve (Paper 2 Style)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# -----------------------------
# 7. Test Behavior
# -----------------------------
temps = []
preds = []

temp = 30
history = [temp] * 5

for _ in range(50):

    temp_pred = predict_temp(history)
    preds.append(temp_pred)

    state = get_state(temp_pred)
    action_idx = np.argmax(Q[state])
    action = [-1, 0, 1][action_idx]

    temp, _ = environment_step(temp, action)
    temps.append(temp)

    history.pop(0)
    history.append(temp)

# -----------------------------
# 8. Plot Results
# -----------------------------
plt.figure()
plt.plot(temps, label="Actual Temp")
plt.plot(preds, label="Predicted Temp")
plt.axhline(25, linestyle='--')
plt.title("Temperature Control with Prediction")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()