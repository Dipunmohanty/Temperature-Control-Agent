import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1. Predictor (LSTM-like)
# -----------------------------
def predict_temp(history):
    weights = np.linspace(1, 2, len(history))
    weights /= weights.sum()
    return np.dot(history, weights)

# -----------------------------
# 2. Environment
# -----------------------------
def environment_step(temp, action, outside_temp=30):
    next_temp = temp + 0.1 * (outside_temp - temp) + action * 0.5
    energy = abs(action)
    return next_temp, energy

# -----------------------------
# 3. State Discretization
# -----------------------------
bins = np.linspace(15, 40, 25)

def get_state(temp):
    return np.digitize(temp, bins)

Q = np.zeros((len(bins)+1, 3))  # Q-table

# -----------------------------
# 4. Parameters
# -----------------------------
alpha = 0.1
gamma = 0.9
epsilon = 0.2
target_temp = 25

rewards = []

print("Training started...")

# -----------------------------
# 5. Training
# -----------------------------
for episode in range(100):

    if episode % 10 == 0:
        print(f"Episode {episode}")

    temp = np.random.uniform(20, 35)
    history = [temp] * 5
    total_reward = 0

    for step in range(20):

        # prediction
        temp_pred = predict_temp(history)

        state = get_state(temp_pred)

        # action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = np.argmax(Q[state])

        action = [-1, 0, 1][action_idx]

        # environment update
        next_temp, energy = environment_step(temp, action)

        # reward
        reward = -(abs(next_temp - target_temp) + 0.1 * energy)
        total_reward += reward

        next_state = get_state(next_temp)

        # Q-learning update
        Q[state, action_idx] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action_idx]
        )

        # update history
        history.pop(0)
        history.append(next_temp)

        temp = next_temp

    rewards.append(total_reward)

# -----------------------------
# 6. Save Learning Graph
# -----------------------------
plt.figure()
plt.plot(rewards)
plt.title("RL Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("learning_curve.png")
plt.close()

# -----------------------------
# 7. Test System
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
# 8. Save Result Graph
# -----------------------------
plt.figure()
plt.plot(temps, label="Actual Temp")
plt.plot(preds, label="Predicted Temp")
plt.axhline(25, linestyle='--', label="Target Temp")
plt.legend()
plt.title("Temperature Control")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.grid()
plt.savefig("temperature_control.png")
plt.close()

print("\nTraining complete!")
print("Graphs saved as:")
print(" - learning_curve.png")
print(" - temperature_control.png")