"""
ODE-based reinforcement learning environment implementation.
"""
import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
from scipy.integrate import solve_ivp


class ODEEnvironment:
    """
    Simple RL environment for ODE systems.
    """
    
    def __init__(
        self,
        model,
        time_step: float = 0.01,
        max_steps: int = 1000,
        reward_function: Optional[Callable] = None,
        integration_method: str = 'RK45',
        state_variables: Optional[List[str]] = None
    ):
        """
        Initialize ODE environment.
        
        Args:
            model: ODE model with step(X, t) method
            time_step: Integration time step
            max_steps: Maximum steps per episode
            reward_function: Function (obs, action, next_obs) -> reward
            integration_method: Scipy integration method ('RK45', 'RK23', 'Radau', 'BDF', 'LSODA')
            state_variables: List of parameter names to include in observations (default: None, uses current state)
        """
        self.model = model
        self.time_step = time_step
        self.max_steps = max_steps
        self.integration_method = integration_method
        self.reward_function = reward_function or (lambda obs, action, next_obs: 0.0)
        self.state_variables = state_variables
        
        # Get initial conditions from model
        self.initial_state = self.model.initial_conditions.copy()
        
        # Current episode state
        self.state = None
        self.time = 0.0
        self.steps = 0
    
    def get_current_observation(self) -> np.ndarray:
        """
        Get current observation based on state_variables configuration.
        
        Returns:
            Current observation vector
        """
        if self.state_variables is None:
            # Default: return the current integrated state
            return self.state.copy() if self.state is not None else self.initial_state.copy()
        else:
            # Return specified parameters from model
            return np.array([self.model.parameters[var] for var in self.state_variables])
    
    def reset(self) -> np.ndarray:
        """Reset to initial state."""
        self.state = self.initial_state.copy()
        self.time = 0.0
        self.steps = 0
        return self.get_current_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take one environment step."""
        prev_obs = self.get_current_observation()
        
        # Integrate ODE for one time step using scipy's solve_ivp
        # This solves: dX/dt = model.step(X, t) from time t to t+dt
        # - fun: the ODE function that returns dX/dt given (t, X)
        # - t_span: time interval to integrate over [t_start, t_end]
        # - y0: initial state at t_start
        # - method: numerical integration method (user-configurable)
        solution = solve_ivp(
            fun=lambda t, X: self.model.step(X, t),
            t_span=(self.time, self.time + self.time_step),
            y0=self.state,
            method=self.integration_method
        )
        
        # Extract the final state after integration
        self.state = solution.y[:, -1]
        self.time += self.time_step
        self.steps += 1
        
        # Get current observation
        next_obs = self.get_current_observation()
        
        # Compute reward
        reward = self.reward_function(prev_obs, action, next_obs)
        
        # Check if done
        done = self.steps >= self.max_steps or np.any(np.abs(self.state) > 1e6)
        
        info = {'time': self.time, 'steps': self.steps}
        
        return next_obs, reward, done, info
    
    def render(self):
        """Print current state."""
        print(f"Step {self.steps}, Time {self.time:.3f}: {self.state}")
    
    def close(self):
        """Close environment."""
        pass
