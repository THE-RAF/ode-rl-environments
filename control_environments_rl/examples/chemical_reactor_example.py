"""
Chemical reactor example using the control-environments-rl framework.

Simple example demonstrating the ChemicalReactor model with reaction A + 2B → 3C.
"""
import numpy as np
from control_environments_rl.src.core.ode_models import ChemicalReactor
from control_environments_rl.src.core.ode_environment import ODEEnvironment


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
        reward_function=product_reward
    )
    
    print(f"Initial: Na={reactor.parameters['Na']}, Nb={reactor.parameters['Nb']}, Nc={reactor.parameters['Nc']}")
    
    # Run simulation
    observation = env.reset()
    total_reward = 0
    
    for step in range(20):  # 2 seconds
        action = np.array([0.0])  # No actions
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            Na, Nb, Nc = observation[0], observation[1], observation[2]
            print(f"t={info['time']:.1f}s: Na={Na:.2f}, Nb={Nb:.2f}, Nc={Nc:.2f}, reward={reward:.1f}")
        
        if done:
            break
    
    print(f"\nFinal: Na={observation[0]:.2f}, Nb={observation[1]:.2f}, Nc={observation[2]:.2f}")
    print(f"Total reward: {total_reward:.1f}")


if __name__ == "__main__":
    main()
