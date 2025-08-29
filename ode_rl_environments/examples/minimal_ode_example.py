"""
Minimal SimpleODE example.
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_rl_environments import ODEEnvironment, ode_models

# Create model and environment
model = ode_models.SimpleODE()
env = ODEEnvironment(model=model, time_step=0.05, max_steps=30)

# ============================================================================== #

# Run episode
obs = env.reset()
rewards = []

for step in range(15):
    action = np.array([0.0])  # No action needed
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done: break

# ============================================================================== #

# Plot results
times = np.arange(len(list(env.model_parameter_history.values())[0])) * 0.05

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
for param, history in env.model_parameter_history.items():
    plt.plot(times, history, label=param, marker='o', markersize=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Simple ODE: dx/dt=y, dy/dt=x')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(rewards)) * 0.05, rewards, 'r-', marker='o', markersize=2)
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Rewards')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
