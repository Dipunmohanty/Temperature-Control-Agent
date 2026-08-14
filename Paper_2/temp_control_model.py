import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
def generate_data(samples=200, timesteps=10):
    X = np.random.rand(samples, timesteps, 3)
    y = np.mean(X[:, :, 0], axis=1, keepdims=True)
    return X, y

X, y = generate_data()

# -----------------------------
# 2. LSTM Model (Prediction)
# -----------------------------
lstm_model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(10, 3)),
    keras.layers.LSTM(50),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

print("Training LSTM...")
history = lstm_model.fit(X, y, epochs=20, verbose=1)

# Plot loss
plt.figure()
plt.plot(history.history['loss'])
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# -----------------------------
# 3. Environment Simulation
# -----------------------------
def environment_step(temp, action):
    # action: -1 cool, 0 maintain, 1 heat
    next_temp = temp + 0.1 * (30 - temp) + action * 0.5
    energy = abs(action)
    return next_temp, energy

# -----------------------------
# 4. DQN Model (Control)
# -----------------------------
dqn_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3)
])

dqn_model.compile(optimizer='adam', loss='mse')

# -----------------------------
# 5. RL Training
# -----------------------------
gamma = 0.9
epsilon = 0.2
target_temp = 25

rewards = []

print("Training RL Controller...")

for episode in range(100):
    temp = np.random.uniform(20, 35)
    total_reward = 0

    for step in range(20):

        state = np.array([[temp]])

        # choose action
        if np.random.rand() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            q_values = dqn_model.predict(state, verbose=0)
            action_idx = np.argmax(q_values)

        action = [-1, 0, 1][action_idx]

        # environment update
        next_temp, energy = environment_step(temp, action)

        # reward
        reward = -(abs(next_temp - target_temp) + 0.1 * energy)
        total_reward += reward

        next_state = np.array([[next_temp]])

        # Q update
        target = reward + gamma * np.max(dqn_model.predict(next_state, verbose=0))
        q_values = dqn_model.predict(state, verbose=0)
        q_values[0][action_idx] = target

        dqn_model.fit(state, q_values, epochs=1, verbose=0)

        temp = next_temp

    rewards.append(total_reward)

# -----------------------------
# 6. Plot RL Learning
# -----------------------------
plt.figure()
plt.plot(rewards)
plt.title("RL Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# -----------------------------
# 7. Test System Behavior
# -----------------------------
temps = []
temp = 30

for _ in range(50):
    state = np.array([[temp]])
    action_idx = np.argmax(dqn_model.predict(state, verbose=0))
    action = [-1, 0, 1][action_idx]

    temp, _ = environment_step(temp, action)
    temps.append(temp)

plt.figure()
plt.plot(temps)
plt.title("Temperature Stabilization")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()