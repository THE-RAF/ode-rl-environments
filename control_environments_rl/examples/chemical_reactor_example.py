"""
Chemical reactor example using the control-environments-rl framework.

Simple example demonstrating the ChemicalReactor model with reaction A + 2B → 3C.
"""
import numpy as np
import matplotlib.pyplot as plt
from control_environments_rl.src.core.ode_models import ChemicalReactor
from control_environments_rl.src.core.ode_environment import ODEEnvironment
from control_environments_rl.src.core.ode_rl_loop import run_rl_episode


def product_reward(model):
    """Simple reward: maximize product C formation."""
    return model.parameters['Nc'] * 10  # 10 points per mole of C


def main():
    """Run simple chemical reactor example."""
    print("Chemical Reactor Example - A + 2B → 3C")
    print("=" * 40)
    
    # Create reactor with initial reactants
    reactor = ChemicalReactor({
        'Na': 2.0,  # 2 moles of A
        'Nb': 4.0,  # 4 moles of B  
        'Nc': 0.0,  # No product initially
        'V': 5.0,   # 5 L reactor
        'k': 1.0    # Rate constant
    })
    
    # Create environment
    env = ODEEnvironment(
        model=reactor,
        time_step=0.1,
        max_steps=50,
        reward_function=product_reward,
        observation_variables=['Na', 'Nb', 'Nc'],
        action_variables=['vai']
    )
    
    print(f"Initial: Na={reactor.parameters['Na']}, Nb={reactor.parameters['Nb']}, Nc={reactor.parameters['Nc']}")
    
    # Define agent vai
    agent = lambda obs: np.random.uniform(0.0, 2.0, size=(1,))

    # Run episode using RL loop
    observations, rewards, total_reward, info = run_rl_episode(
        env=env,
        agent_function=agent,
        max_steps=20,
        verbose=False
    )
    
    # Final results summary
    final_obs = info['final_observation']
    print(f"Final: Na={final_obs[0]:.2f}, Nb={final_obs[1]:.2f}, Nc={final_obs[2]:.2f}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Episode completed in {info['steps']} steps")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Create time arrays - parameter history includes initial state
    param_times = np.arange(len(list(env.model_parameter_history.values())[0])) * 0.1
    reward_times = np.arange(len(rewards)) * 0.1  # time_step = 0.1
    
    # Plot 1: Parameter history over time
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for param_name, history in env.model_parameter_history.items():
        if param_name in ['Na', 'Nb', 'Nc']:  # Only plot concentration parameters
            plt.plot(param_times, history, label=f'{param_name} (mol)', marker='o', markersize=3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mol)')
    plt.title('Chemical Reactor - A + 2B → 3C')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rewards over time
    plt.subplot(1, 2, 2)
    plt.plot(reward_times, rewards, 'r-', label='Reward', marker='o', markersize=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
