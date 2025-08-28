"""
Simple ODE example using the control-environments-rl framework.

This example demonstrates how to use the SimpleODE model (dx/dt = y, dy/dt = x)
with the RL environment. The system is naturally unstable and will grow
exponentially without control.
"""
import numpy as np
from control_environments_rl.src.core.ode_models import SimpleODE
from control_environments_rl.src.core.ode_environment import ODEEnvironment
from control_environments_rl.src.utils.visualization import (
    plot_episode_results, 
    plot_phase_portrait, 
    plot_reward_analysis
)


def custom_reward_function(obs, action, next_obs):
    """
    Custom reward function: minimize distance from origin.
    
    Args:
        obs: Previous state [x, y]
        action: Action taken (unused in this example)
        next_obs: Current state [x, y]
        
    Returns:
        Negative distance from origin (higher reward = closer to origin)
    """
    distance = np.sqrt(np.sum(next_obs**2))
    return -distance


def main():
    """Run simple ODE example."""
    print("Simple ODE Environment Example")
    print("System: dx/dt = y, dy/dt = x")
    print("-" * 40)
    
    # Create the ODE model
    model = SimpleODE()  # Uses default parameters: x=1.0, y=0.5
    print(f"Initial conditions: x={model.parameters['x']}, y={model.parameters['y']}")
    
    # Create the RL environment
    env = ODEEnvironment(
        model=model,
        time_step=0.01,
        max_steps=100,
        reward_function=custom_reward_function,
        integration_method='RK45'
    )
    
    # Reset environment
    observation = env.reset()
    print(f"Initial observation: {observation}")
    
    # Run episode with random actions
    total_reward = 0
    states_history = [observation.copy()]
    rewards_history = []
    times_history = [0.0]
    actions_history = []
    
    print("\nRunning episode with random actions:")
    
    for step in range(20):  # Run for 20 steps
        # Random action (not used in reward function, but required by interface)
        action = np.random.uniform(-0.1, 0.1, size=2)
        
        # Take step
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        # Store history for plotting
        states_history.append(observation.copy())
        rewards_history.append(reward)
        times_history.append(info['time'])
        actions_history.append(action.copy())
        
        # Print every 5 steps
        if step % 5 == 0:
            obs_str = f"[{observation[0]:.3f}, {observation[1]:.3f}]"
            print(f"Step {step}: obs={obs_str}, reward={reward:.3f}, time={info['time']:.3f}")
        
        if done:
            print(f"Episode finished early at step {step}")
            break
    
    print(f"\nEpisode completed!")
    print(f"Final observation: {observation}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final time: {info['time']:.3f}")
    
    # Demonstrate the unstable nature
    print(f"\nSystem growth: Initial magnitude = {np.linalg.norm(env.initial_state):.3f}")
    print(f"                Final magnitude = {np.linalg.norm(observation):.3f}")
    print(f"Growth factor: {np.linalg.norm(observation) / np.linalg.norm(env.initial_state):.2f}x")
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # Plot complete episode results (align arrays)
    plot_episode_results(
        states=states_history[1:],  # Skip initial state to match rewards
        rewards=rewards_history,
        times=times_history[1:],    # Skip initial time to match rewards
        actions=actions_history,
        state_labels=['x', 'y'],
        title='Simple ODE System'
    )
    
    # Plot phase portrait (trajectory in state space)
    plot_phase_portrait(
        states=states_history,
        title='Simple ODE Phase Portrait',
        state_labels=['x', 'y']
    )
    
    # Plot detailed reward analysis
    plot_reward_analysis(
        rewards=rewards_history,
        times=times_history[1:],  # Skip initial time (no reward yet)
        title='Simple ODE Rewards'
    )


if __name__ == "__main__":
    main()
