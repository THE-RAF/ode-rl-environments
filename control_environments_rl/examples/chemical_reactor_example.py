"""
Chemical reactor example using the control-environments-rl framework.

This example demonstrates the ChemicalReactor model with reaction A + 2B → 3C
in a continuous stirred-tank reactor (CSTR). Shows different operating scenarios:
batch mode, continuous feeding, and product optimization.
"""
import numpy as np
from control_environments_rl.src.core.ode_models import ChemicalReactor
from control_environments_rl.src.core.ode_environment import ODEEnvironment
from control_environments_rl.src.utils.visualization import (
    plot_episode_results, 
    plot_phase_portrait, 
    plot_reward_analysis,
    plot_state_evolution
)


def product_maximization_reward(obs, action, next_obs):
    """
    Reward function: maximize product formation (moles of C).
    
    Args:
        obs: Previous state [Na, Nb, Nc]
        action: Action taken (unused in this example)
        next_obs: Current state [Na, Nb, Nc]
        
    Returns:
        Reward based on product formation rate and total product
    """
    Na, Nb, Nc = next_obs[0], next_obs[1], next_obs[2]
    
    # Reward for product formation (moles of C)
    product_reward = Nc * 10  # 10 points per mole of C
    
    # Penalty for unused reactants (encourage conversion)
    unused_penalty = -(Na + Nb) * 0.5  # Small penalty for leftover reactants
    
    # Bonus for balanced stoichiometry (A:B ratio close to 1:2)
    if Nb > 0 and Na > 0:
        ratio = Nb / (2 * Na)  # Ideal ratio is 1.0
        balance_bonus = 5.0 * np.exp(-abs(ratio - 1.0))  # Bonus peaks at ratio = 1
    else:
        balance_bonus = 0.0
    
    return product_reward + unused_penalty + balance_bonus


def main():
    """Run chemical reactor examples."""
    print("Chemical Reactor Environment Example")
    print("Reaction: A + 2B → 3C")
    print("=" * 50)
    
    # Scenario 1: Batch reactor with initial charge
    print("\n1. BATCH REACTOR SCENARIO")
    print("-" * 30)
    
    # Create reactor with initial reactants (batch mode)
    reactor_batch = ChemicalReactor({
        'Na': 2.0,    # 2 moles of A initially
        'Nb': 4.0,    # 4 moles of B initially (stoichiometric)
        'Nc': 0.0,    # No product initially
        'vai': 0.0,   # No inlet flows (batch)
        'vbi': 0.0,
        'vci': 0.0,
        'V': 5.0,     # 5 L reactor
        'k': 1.0      # Moderate reaction rate
    })
    
    print(f"Initial charge: Na={reactor_batch.parameters['Na']} mol, "
          f"Nb={reactor_batch.parameters['Nb']} mol, Nc={reactor_batch.parameters['Nc']} mol")
    print(f"Reactor volume: {reactor_batch.parameters['V']} L")
    print(f"Rate constant: {reactor_batch.parameters['k']} L²/(mol²·s)")
    
    # Create environment
    env_batch = ODEEnvironment(
        model=reactor_batch,
        time_step=0.1,
        max_steps=100,
        reward_function=product_maximization_reward,
        integration_method='RK45'
    )
    
    # Run batch simulation
    observation = env_batch.reset()
    print(f"Initial state: [Na={observation[0]:.2f}, Nb={observation[1]:.2f}, Nc={observation[2]:.2f}] mol")
    
    # Storage for plotting
    states_batch = [observation.copy()]
    rewards_batch = []
    times_batch = [0.0]
    
    print("\nBatch reaction progress:")
    
    for step in range(50):  # Run for 5 seconds
        # No control actions in batch mode
        action = np.array([0.0, 0.0, 0.0])
        
        observation, reward, done, info = env_batch.step(action)
        
        # Store data
        states_batch.append(observation.copy())
        rewards_batch.append(reward)
        times_batch.append(info['time'])
        
        # Print every 10 steps
        if step % 10 == 0:
            Na, Nb, Nc = observation[0], observation[1], observation[2]
            Ca = Na / reactor_batch.parameters['V']
            Cb = Nb / reactor_batch.parameters['V']
            Cc = Nc / reactor_batch.parameters['V']
            conversion = (2.0 - Na) / 2.0 * 100  # % conversion of A
            
            print(f"t={info['time']:.1f}s: Na={Na:.2f}, Nb={Nb:.2f}, Nc={Nc:.2f} mol | "
                  f"Ca={Ca:.3f}, Cb={Cb:.3f}, Cc={Cc:.3f} M | Conv={conversion:.1f}%")
        
        if done:
            break
    
    # Final analysis
    final_Na, final_Nb, final_Nc = observation[0], observation[1], observation[2]
    final_conversion = (2.0 - final_Na) / 2.0 * 100
    yield_C = final_Nc / (3 * (2.0 - final_Na)) * 100 if final_Na < 2.0 else 0  # Theoretical yield
    
    print(f"\nBatch Results:")
    print(f"  Final moles: Na={final_Na:.3f}, Nb={final_Nb:.3f}, Nc={final_Nc:.3f}")
    print(f"  Conversion of A: {final_conversion:.1f}%")
    print(f"  Yield of C: {yield_C:.1f}%")
    print(f"  Total reward: {sum(rewards_batch):.1f}")
    
    # Scenario 2: Continuous reactor with feeding
    print(f"\n2. CONTINUOUS REACTOR SCENARIO")
    print("-" * 35)
    
    # Create continuous reactor
    reactor_continuous = ChemicalReactor({
        'Na': 0.0,    # Start empty
        'Nb': 0.0,
        'Nc': 0.0,
        'vai': 0.1,   # Feed A at 0.1 L/s
        'vbi': 0.2,   # Feed B at 0.2 L/s (excess for better conversion)
        'vci': 0.0,   # No C feed
        'V': 10.0,    # Larger reactor (10 L)
        'Cai': 2.0,   # 2 M A in feed
        'Cbi': 3.0,   # 3 M B in feed
        'Cci': 0.0,   # No C in feed
        'k': 0.5      # Lower rate constant for continuous operation
    })
    
    print(f"Feed rates: A={reactor_continuous.parameters['vai']} L/s, "
          f"B={reactor_continuous.parameters['vbi']} L/s")
    print(f"Feed concentrations: Cai={reactor_continuous.parameters['Cai']} M, "
          f"Cbi={reactor_continuous.parameters['Cbi']} M")
    
    # Create environment
    env_continuous = ODEEnvironment(
        model=reactor_continuous,
        time_step=0.2,
        max_steps=150,
        reward_function=product_maximization_reward,
        integration_method='RK45'
    )
    
    # Run continuous simulation
    observation = env_continuous.reset()
    
    # Storage for plotting
    states_continuous = [observation.copy()]
    rewards_continuous = []
    times_continuous = [0.0]
    
    print("\nContinuous reactor startup:")
    
    for step in range(75):  # Run for 15 seconds
        # Small random disturbances in feed rates
        action = np.random.uniform(-0.01, 0.01, size=3)
        
        observation, reward, done, info = env_continuous.step(action)
        
        # Store data
        states_continuous.append(observation.copy())
        rewards_continuous.append(reward)
        times_continuous.append(info['time'])
        
        # Print every 15 steps
        if step % 15 == 0:
            Na, Nb, Nc = observation[0], observation[1], observation[2]
            Ca = Na / reactor_continuous.parameters['V']
            Cb = Nb / reactor_continuous.parameters['V']
            Cc = Nc / reactor_continuous.parameters['V']
            
            print(f"t={info['time']:.1f}s: Na={Na:.1f}, Nb={Nb:.1f}, Nc={Nc:.1f} mol | "
                  f"Ca={Ca:.2f}, Cb={Cb:.2f}, Cc={Cc:.2f} M | Reward={reward:.1f}")
        
        if done:
            break
    
    # Final continuous analysis
    print(f"\nContinuous Results:")
    print(f"  Final moles: Na={observation[0]:.2f}, Nb={observation[1]:.2f}, Nc={observation[2]:.2f}")
    print(f"  Final concentrations: Ca={observation[0]/10:.3f}, Cb={observation[1]/10:.3f}, Cc={observation[2]/10:.3f} M")
    print(f"  Average reward: {np.mean(rewards_continuous[-10:]):.1f} (last 10 steps)")
    
    # Create visualizations
    print(f"\n3. GENERATING VISUALIZATION PLOTS")
    print("-" * 35)
    
    # Plot batch reactor results
    plot_episode_results(
        states=states_batch[1:],
        rewards=rewards_batch,
        times=times_batch[1:],
        actions=None,  # No actions in batch mode
        state_labels=['Na (mol)', 'Nb (mol)', 'Nc (mol)'],
        title='Batch Chemical Reactor - A + 2B → 3C'
    )
    
    # Plot continuous reactor results
    plot_episode_results(
        states=states_continuous[1:],
        rewards=rewards_continuous,
        times=times_continuous[1:],
        actions=None,  # Actions are small disturbances
        state_labels=['Na (mol)', 'Nb (mol)', 'Nc (mol)'],
        title='Continuous Chemical Reactor - A + 2B → 3C'
    )
    
    # Plot detailed state evolution for batch
    plot_state_evolution(
        states=states_batch,
        times=times_batch,
        state_labels=['Na - Reactant A (mol)', 'Nb - Reactant B (mol)', 'Nc - Product C (mol)'],
        title='Batch Reactor - Species Evolution'
    )
    
    # Plot 3D phase space (if we had 3D plotting, we'd use all three states)
    # For now, plot Na vs Nc (reactant vs product)
    batch_array = np.array(states_batch)
    plot_phase_portrait(
        states=[[s[0], s[2]] for s in states_batch],  # Na vs Nc
        title='Batch Reactor: Reactant A vs Product C',
        state_labels=['Na - Reactant A (mol)', 'Nc - Product C (mol)']
    )
    
    # Reward analysis for both scenarios
    plot_reward_analysis(
        rewards=rewards_batch,
        times=times_batch[1:],
        title='Batch Reactor Performance',
        window_size=5
    )
    
    plot_reward_analysis(
        rewards=rewards_continuous,
        times=times_continuous[1:],
        title='Continuous Reactor Performance',
        window_size=10
    )
    
    # Summary comparison
    print(f"\n4. SCENARIO COMPARISON")
    print("-" * 20)
    print(f"Batch Reactor:")
    print(f"  Product formed: {final_Nc:.2f} mol C")
    print(f"  Time to completion: {times_batch[-1]:.1f} s")
    print(f"  Peak reward: {max(rewards_batch):.1f}")
    
    print(f"\nContinuous Reactor:")
    print(f"  Steady-state product: {observation[2]:.2f} mol C")
    print(f"  Startup time: {times_continuous[-1]:.1f} s")
    print(f"  Steady-state reward: {np.mean(rewards_continuous[-10:]):.1f}")
    
    print(f"\nBoth scenarios demonstrate the rich dynamics of chemical reactor control!")


if __name__ == "__main__":
    main()
