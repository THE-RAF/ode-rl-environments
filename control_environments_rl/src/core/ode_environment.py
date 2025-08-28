"""
ODE-based reinforcement learning environment implementation.
"""
import numpy as np
from typing import Dict, Callable, Optional, Tuple
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
        integration_method: str = 'RK45'
    ):
        """
        Initialize ODE environment.
        
        Args:
            model: ODE model with step(X, t) method
            time_step: Integration time step
            max_steps: Maximum steps per episode
            reward_function: Function (obs, action, next_obs) -> reward
            integration_method: Scipy integration method ('RK45', 'RK23', 'Radau', 'BDF', 'LSODA')
        """
        self.model = model
        self.time_step = time_step
        self.max_steps = max_steps
        self.reward_function = reward_function or self._default_reward
        self.integration_method = integration_method
        
        # Get initial state from model parameters
        self.initial_state = self._get_initial_state()
        
        # Current episode state
        self.state = None
        self.time = 0.0
        self.steps = 0
    
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state from model parameters."""
        params = self.model.parameters
        if 'x' in params and 'y' in params:
            return np.array([params['x'], params['y']])
        elif 'Tv' in params and 'Tj' in params:
            return np.array([params['Tv'], params['Tj']])
        else:
            raise ValueError("Could not extract initial state from model")
    
    def _default_reward(self, obs, action, next_obs) -> float:
        """Default reward: return 0 (no reward signal)."""
        return 0.0
    
    def reset(self) -> np.ndarray:
        """Reset to initial state."""
        self.state = self.initial_state.copy()
        self.time = 0.0
        self.steps = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take one environment step."""
        prev_state = self.state.copy()
        
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
        
        # Compute reward
        reward = self.reward_function(prev_state, action, self.state)
        
        # Check if done
        done = self.steps >= self.max_steps or np.any(np.abs(self.state) > 1e6)
        
        info = {'time': self.time, 'steps': self.steps}
        
        return self.state.copy(), reward, done, info
    
    def render(self):
        """Print current state."""
        print(f"Step {self.steps}, Time {self.time:.3f}: {self.state}")
    
    def close(self):
        """Close environment."""
        pass
