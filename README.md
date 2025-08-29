# ODE-RL Environments - ODE Systems as Reinforcement Learning Environments

A clean, professional framework for transforming ordinary differential equation (ODE) systems into reinforcement learning environments, compatible with standard RL interfaces like OpenAI Gym.

## Features

- **Simple API**: Single `create_ode_environment()` function with sensible defaults
- **Standard RL Interface**: Compatible with OpenAI Gym - `reset()`, `step()`, `render()`, `close()`
- **Built-in ODE Models**: Theoretical 2x2 system and heated tank examples included
- **Custom Model Support**: Easy integration of your own ODE systems
- **Integrated ODE Solving**: Built-in numerical integration using scipy
- **Flexible Reward Functions**: Define custom reward logic for your control objectives

## Quick Start

```python
from ode_rl_environments import create_ode_environment
from ode_rl_environments.models import Theoretical2x2

# Define a reward function for your control objective
def reward_function(obs, action, next_obs):
    # Example: minimize distance from origin
    return -np.sum(next_obs**2)

# Create ODE model
model = Theoretical2x2(parameters={
    'x0': 1.0,  # Initial state x0
    'x1': 1.0,  # Initial state x1
    'u0': 0.0,  # Control input u0
    'u1': 0.0   # Control input u1
})

# Create RL environment
env = create_ode_environment(
    model=model,
    time_step=0.01,
    max_episode_steps=1000,
    reward_function=reward_function
)

# Standard RL loop
observation = env.reset()
total_reward = 0

for step in range(100):
    # Your RL agent selects action (control inputs)
    action = your_agent.select_action(observation)
    
    # Environment steps forward in time
    observation, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        break

print(f"Episode completed with total reward: {total_reward}")
```

## Built-in Models

### Theoretical 2x2 System
Linear system with state feedback:
```python
from ode_rl_environments.models import Theoretical2x2

model = Theoretical2x2(parameters={
    'x0': 1.0,   # Initial condition for state 1
    'x1': 1.0,   # Initial condition for state 2
    'u0': 0.0,   # Control input 1
    'u1': 0.0    # Control input 2
})
```

System equations:
- State: `X = [x0, x1]`
- Control: `U = [u0, u1]` (modified by RL actions)
- Dynamics: `dX/dt = A*X + B*U`

### Heated Tank System
Non-linear tank temperature control:
```python
from ode_rl_environments.models import HeatedTank

model = HeatedTank(parameters={
    'Tv': 350.0,    # Initial tank temperature
    'Tj': 300.0,    # Initial jacket temperature
    'F': 5.0,       # Flow rate
    'Fj': 2.0,      # Jacket flow rate
    'T0': 300.0,    # Feed temperature
    'Tj0': 350.0,   # Jacket feed temperature
    'V': 100.0,     # Tank volume
    'Vj': 10.0,     # Jacket volume
    'rho': 1000.0,  # Density
    'rhoj': 1000.0, # Jacket density
    'cp': 4.18,     # Heat capacity
    'cpj': 4.18,    # Jacket heat capacity
    'U': 500.0,     # Heat transfer coefficient
    'A': 2.0        # Heat transfer area
})
```

## Custom ODE Models

Create your own ODE systems by inheriting from the base class:

```python
from ode_rl_environments.models import ODEModel
import numpy as np

class MyCustomODE(ODEModel):
    def __init__(self, parameters):
        self.parameters = parameters
        # Set up your system matrices, constants, etc.
    
    def step(self, X, t):
        # Implement your ODE: dX/dt = f(X, t, U)
        # Access current state: X[0], X[1], ...
        # Access control inputs: self.parameters['u0'], etc.
        
        dXdt = np.zeros_like(X)
        # Your ODE equations here...
        return dXdt
    
    @property
    def initial_conditions(self):
        return np.array([self.parameters['x0'], self.parameters['x1']])

# Use your custom model
my_model = MyCustomODE(parameters={...})
env = create_ode_environment(model=my_model, ...)
```

## API Reference

### create_ode_environment()

Main function to create an RL environment from an ODE model:

```python
env = create_ode_environment(
    model,                    # ODEModel instance
    time_step=0.01,          # Integration time step (seconds)
    max_episode_steps=1000,  # Maximum steps per episode
    reward_function=None,    # Custom reward function
    action_bounds=[-1, 1],   # Action space bounds
    solver_method='RK45'     # ODE solver method
)
```

**Parameters:**
- `model`: An instance of `ODEModel` (built-in or custom)
- `time_step`: Time step for numerical integration
- `max_episode_steps`: Maximum number of steps before episode termination
- `reward_function`: Function `(obs, action, next_obs) -> reward`
- `action_bounds`: `[min, max]` bounds for action space
- `solver_method`: Scipy solver method ('RK45', 'RK23', 'Radau', etc.)

**Returns:**
- RL environment with standard interface (`reset()`, `step()`, `render()`, `close()`)

### Environment Interface

The created environment follows the standard RL interface:

```python
# Reset environment to initial state
observation = env.reset()

# Take action and get next state
observation, reward, done, info = env.step(action)

# Render current state (optional)
env.render()

# Clean up resources
env.close()
```

**Observation Space:** Current system state vector (X)  
**Action Space:** Control input modifications (added to current control inputs)  
**Reward:** Returned by your custom reward function  
**Done:** `True` when episode reaches max steps or custom termination condition  
**Info:** Dictionary with additional information (time, raw state, etc.)  

## Installation

```bash
# Install from source (recommended for development)
git clone https://github.com/THE-RAF/control-environments-rl.git
cd control-environments-rl
pip install -e .

# Or install from GitHub
pip install git+https://github.com/THE-RAF/control-environments-rl.git
```

## Examples

Check out the `examples/` directory for complete usage examples:

- `theoretical_2x2_example.py` - Basic linear system control
- `heated_tank_example.py` - Non-linear temperature control
- `custom_ode_example.py` - Creating your own ODE model

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0 (for visualization)

## License

MIT License - see LICENSE file for details.

## Contributing

This project follows clean, professional coding standards. Please ensure your contributions:

1. Include comprehensive type hints
2. Have detailed docstrings
3. Follow the established architecture patterns
4. Include examples for new features
5. Pass all validation checks

---

Transform your differential equations into RL environments with just a few lines of code!
