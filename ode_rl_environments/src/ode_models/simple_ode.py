"""
Simple ODE model implementation.
"""
import numpy as np
from typing import Dict, Any


class SimpleODE:
    """
    Simple coupled ODE system.
    
    System equations:
        dx/dt = y
        dy/dt = x
    
    This creates an unstable system where both x and y grow exponentially.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the simple ODE system.
        
        Args:
            parameters: Dictionary with keys:
                - 'x': Initial condition for x (default: 1.0)
                - 'y': Initial condition for y (default: 0.5)
        """
        if parameters is None:
            parameters = {}
        
        # Set default values
        self.parameters = {
            'x': parameters.get('x', 1.0),
            'y': parameters.get('y', 0.5)
        }
        
        # Set initial conditions
        self.initial_conditions = np.array([self.parameters['x'], self.parameters['y']])
    
    def step(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Compute system dynamics: dx/dt = y, dy/dt = x
        
        Args:
            X: State vector [x, y]
            t: Current time (unused in this system)
            
        Returns:
            dX/dt: State derivative [dx/dt, dy/dt]
        """
        x, y = X[0], X[1]
        
        # Update current state in parameters
        self.parameters['x'] = x
        self.parameters['y'] = y
        
        # Simple coupled system
        dx_dt = y
        dy_dt = x
        
        return np.array([dx_dt, dy_dt])
