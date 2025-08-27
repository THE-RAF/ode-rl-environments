# Implementation Plan for `control-environments-rl`

## Core Concept
Transform ODE systems into RL environments by:
1. Taking ODE models that follow the established `step(X, t)` pattern
2. Wrapping them with RL environment interface (`reset()`, `step(action)`, `observation`, `reward`, `done`)
3. Using ODE solvers integrated directly in the environment
4. Providing a clean API similar to existing libraries

## Architecture Overview

```
control-environments-rl/
├── pyproject.toml
├── README.md
├── LICENSE
├── MANIFEST.in
├── control_environments_rl/
│   ├── __init__.py                    # Main API entry point
│   ├── examples/
│   │   ├── theoretical_2x2_example.py
│   │   ├── heated_tank_example.py
│   │   └── custom_ode_example.py
│   └── src/
│       ├── core/
│       │   ├── ode_environment.py     # Main RL environment class (includes integration)
│       │   └── ode_models.py          # Built-in ODE models + base class
│       └── utils/
│           ├── config_validation.py   # Parameter validation
│           └── visualization.py       # Plotting utilities
```

## Main API Design

Following the pattern of ultra-clean single-function interfaces:

```python
from control_environments_rl import create_ode_environment
from control_environments_rl.models import Theoretical2x2, HeatedTank

# Usage with built-in models
model = Theoretical2x2(parameters={'x0': 1, 'x1': 1, 'u0': 1, 'u1': 1})
env = create_ode_environment(
    model=model,
    time_step=0.01,
    max_episode_steps=1000,
    reward_function=custom_reward_fn
)

# Usage with custom ODE models
custom_model = MyCustomODE(parameters={...})
env = create_ode_environment(
    model=custom_model,
    time_step=0.01,
    max_episode_steps=1000,
    reward_function=custom_reward_fn
)

# Standard RL loop works immediately
observation = env.reset()
for step in range(100):
    action = agent.select_action(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break
```

## Core Components

### 1. ODEEnvironment Class (`src/core/ode_environment.py`)
- Implements standard RL environment interface
- Wraps ODE models with **integrated** ODE solving (no separate integration.py)
- Handles observation/action/reward logic
- Manages episode lifecycle
- Contains scipy ODE solver functionality directly

### 2. ODE Models (`src/core/ode_models.py`)
- `ODEModel` abstract base class for custom models
- `Theoretical2x2` class (from references)
- `HeatedTank` class (from references)
- All models follow the pattern: `__init__(parameters)`, `step(X, t)`, `initial_conditions`

### 3. Configuration & Validation (`src/utils/config_validation.py`)
- Parameter validation following established patterns
- Clear error messages for invalid configurations

### 4. Visualization (`src/utils/visualization.py`)
- Plotting utilities for system states, rewards, actions
- Episode trajectory visualization

## Key Design Decisions

### 1. Simple ODE Model Interface
All ODE models follow the established pattern:
```python
class ODEModel:
    def __init__(self, parameters):
        self.parameters = parameters
        # Set up initial conditions, system matrices, etc.
    
    def step(self, X, t):
        # Return dXdt given current state X and time t
        return dXdt
    
    @property
    def initial_conditions(self):
        # Return initial state vector
        return initial_state_array
```

### 2. Flexible Reward Functions
Users provide reward functions that take `(observation, action, next_observation)`:
```python
def reward_function(obs, action, next_obs):
    # Custom reward logic based on state transitions
    return reward_value
```

### 3. Direct Model Passing
- No model selection by string - pass model objects directly
- Keeps the API simple and flexible
- Users instantiate their own models with their own parameters

### 4. Integrated ODE Solving
- No separate `integration.py` file
- ODE solving functionality built directly into `ODEEnvironment`
- Uses scipy's `solve_ivp` or similar for single time step integration

### 5. Standard RL Environment Interface
```python
class ODEEnvironment:
    def reset(self) -> observation
    def step(self, action) -> (observation, reward, done, info)
    def render(self, mode='human'): ...
    def close(self): ...
```

## Implementation Phases

### Phase 1: Core Framework
1. `ODEEnvironment` base class with integrated ODE solving
2. `ODEModel` abstract base class
3. Basic parameter validation
4. Simple built-in model (Theoretical2x2)

### Phase 2: Enhanced Features
1. `HeatedTank` model implementation
2. Visualization utilities
3. More sophisticated reward function helpers
4. Better error handling and edge cases

### Phase 3: Examples & Documentation
1. Complete example files for each model type
2. README with quick start guide
3. Comprehensive docstrings
4. Package setup and testing

## Architecture Principles Followed

✅ **Single main function**: `create_ode_environment()`  
✅ **Sensible defaults**: Time step, solver method, episode length  
✅ **Clean return objects**: Standard RL environment interface  
✅ **Comprehensive validation**: Parameter checking with clear errors  
✅ **Layered architecture**: Core/utils separation  
✅ **Type hints throughout**: Full typing support  
✅ **Professional packaging**: Complete pyproject.toml setup  
✅ **Examples included**: Simple to complex usage patterns  
✅ **Direct object passing**: No string-based model selection  
✅ **Integrated functionality**: ODE solving built into environment  

## Notes
- Keep implementation simple and focused
- Follow existing library patterns exactly
- Prioritize ease of use over feature complexity
- Maintain clean separation between ODE models and RL environment logic
