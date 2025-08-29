"""
Chemical reactor example using the control-environments-rl framework.
"""
import numpy as np
import matplotlib.pyplot as plt
from control_environments_rl import ODEEnvironment, ode_models


def product_reward(model):
    """Simple reward: maximize product C formation."""
    return model.parameters['Nc'] * 10

# Create reactor and environment
reactor = ode_models.ChemicalReactor()
env = ODEEnvironment(
    model=reactor,
    time_step=0.1,
    max_steps=50,
    reward_function=product_reward,
    observation_variables=['Na', 'Nb', 'Nc'],
    action_variables=['vai', 'vbi', 'vci']
)

# Define agent
agent = lambda obs: np.random.uniform(0.0, 2.0, size=(3,))

# ============================================================================== #

# Run RL episode with traditional loop
obs = env.reset()
observations = [obs.copy()]
rewards = []
total_reward = 0.0

for step in range(20):  # Run for 20 steps
    # Agent selects action
    action = agent(obs)
    
    # Environment step
    obs, reward, done, info = env.step(action)
    
    # Store results
    observations.append(obs.copy())
    rewards.append(reward)
    total_reward += reward
    
    # Check if episode is done
    if done:
        break

print(f"Episode finished after {step + 1} steps with total reward: {total_reward:.1f}")

# ============================================================================== #

# Plot results using parameter history
param_times = np.arange(len(list(env.model_parameter_history.values())[0])) * 0.1
reward_times = np.arange(len(rewards)) * 0.1

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for param_name, history in env.model_parameter_history.items():
    if param_name in ['Na', 'Nb', 'Nc']:
        plt.plot(param_times, history, label=f'{param_name} (mol)', marker='o', markersize=3)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol)')
plt.title('Chemical Reactor - Inline RL Loop')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(reward_times, rewards, 'r-', label='Reward', marker='o', markersize=3)
plt.xlabel('Time (s)')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
