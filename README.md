# ODE-RL Environments

A clean, professional framework for transforming ordinary differential equation (ODE) systems into reinforcement learning environments.

## Features

- **Simple API**: Create custom ODE models and wrap them with `ODEEnvironment` 
- **Standard RL Interface**: Compatible with OpenAI Gym - `reset()`, `step()`, `render()`, `close()`
- **Built-in ODE Models**: SimpleODE, ChemicalReactor, and ControlledTank examples included
- **Custom Model Support**: Easy integration of your own ODE systems
- **Flexible Observations/Actions**: Configure which model parameters become RL observations and actions
- **Parameter History Tracking**: Automatic logging for visualization and analysis
- **Integrated ODE Solving**: Built-in numerical integration using scipy with configurable methods

## Quick Start

```python
from ode_rl_environments import ODEEnvironment, ode_models
import numpy as np

# Define a reward function
def product_reward(model):
    return model.parameters['Nc'] * 10  # Maximize product C

# Create model and environment
reactor = ode_models.ChemicalReactor()
env = ODEEnvironment(
    model=reactor,
    time_step=0.1,
    reward_function=product_reward,
    observation_variables=['Na', 'Nb', 'Nc'],
    action_variables=['vai', 'vbi', 'vci']
)

# Standard RL loop
obs = env.reset()
for step in range(20):
    action = np.random.uniform(0.0, 2.0, size=(3,))  # Random agent
    obs, reward, done, info = env.step(action)
    if done: break

# Visualize results
import matplotlib.pyplot as plt
times = np.arange(len(env.model_parameter_history['Nc'])) * 0.1
plt.plot(times, env.model_parameter_history['Nc'], label='Product C')
plt.show()
```

## Installation

Clone and install:

```bash
pip install git+https://github.com/THE-RAF/ode-rl-environments.git
```

## Creating Custom ODE Models

To create your own ODE system, implement a class with `step(X, t)` method:

```python
import numpy as np

class MyODEModel:
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        
        # Define model parameters
        self.parameters = {
            'x': parameters.get('x', 1.0),
            'y': parameters.get('y', 0.5),
            'u': parameters.get('u', 0.0)  # Control input
        }
        
        # Define initial conditions
        self.initial_conditions = np.array([self.parameters['x'], self.parameters['y']])
    
    def step(self, X, t):
        """Compute dX/dt given current state X and time t"""
        x, y = X[0], X[1]
        
        # Update current state in parameters
        self.parameters['x'] = x
        self.parameters['y'] = y
        
        # Define your ODE: dx/dt = f(x,y,u,t)
        u = self.parameters['u']  # Get control input
        dx_dt = -x + u
        dy_dt = x - y
        
        return np.array([dx_dt, dy_dt])

# Use your custom model
model = MyODEModel()
env = ODEEnvironment(
    model=model,
    observation_variables=['x', 'y'],
    action_variables=['u']
)
```

## API Reference

### ODEEnvironment

Main environment class for wrapping ODE models:

- `model`: ODE model object with `step(X, t)` method and `initial_conditions`
- `time_step`: Integration time step (default: 0.01)
- `max_steps`: Maximum steps per episode (default: 1000)
- `reward_function`: Function `(model) -> float` for computing rewards (default: returns 0)
- `integration_method`: Scipy method - 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA' (default: 'RK45')
- `observation_variables`: List of parameter names to include in observations (default: uses state vector)
- `action_variables`: List of parameter names that actions will modify (default: no actions)

### Built-in Models

**SimpleODE**: Coupled system with dx/dt = y, dy/dt = x
```python
model = ode_models.SimpleODE(parameters={'x': 1.0, 'y': 0.5})
```

**ChemicalReactor**: CSTR with reaction A + 2B → 3C
```python
model = ode_models.ChemicalReactor(parameters={
    'Na': 0.0, 'Nb': 0.0, 'Nc': 0.0,  # Initial moles
    'vai': 0.0, 'vbi': 0.0, 'vci': 0.0  # Flow rates (actions)
})
```

**ControlledTank**: Variable height tank with mass balance
```python
model = ode_models.ControlledTank(parameters={
    'm': 10.0,      # Initial mass
    'vi': 0.0,      # Inlet flow (action)
    'hSetpoint': 5.0  # Height target
})
```

## Standard RL Loop

The environment follows OpenAI Gym interface:

```python
# Episode setup
obs = env.reset()  # Returns initial observation
total_reward = 0

for step in range(max_steps):
    # Agent decides action (not implemented - up to you!)
    action = your_rl_agent(obs)
    
    # Environment step
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        break

# Access parameter history for analysis
history = env.model_parameter_history
```

**Note**: This library provides the RL environment only. The RL agent implementation is outside the scope of this module and left to the user.

## Examples

Run the included examples:

```bash
python -m ode_rl_environments.examples.minimal_ode_example
python -m ode_rl_environments.examples.chemical_reactor_example
python -m ode_rl_environments.examples.controlled_tank_example
```

Each example demonstrates:
- Model creation and environment setup
- Custom reward functions
- Parameter history visualization
- Different observation/action configurations

## License

This ODE-RL environments module is provided as-is for educational and research purposes.
