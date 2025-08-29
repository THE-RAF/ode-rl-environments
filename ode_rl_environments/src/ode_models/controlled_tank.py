"""
Variable height tank model implementation.
"""
import numpy as np
from typing import Dict, Any


class ControlledTank:
    """
    Simple variable height tank with mass balance.
    
    System equations:
        mi = vi * rho
        h = m / (rho * A)
        h = max(h, 0)
        mo = beta * sqrt(h)
        v = mo / rho
        dm/dt = mi - mo
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the tank system.
        
        Args:
            parameters: Dictionary with keys:
                - 'm': Initial mass (default: 10.0)
                - 'vi': Inlet flow rate (default: 1.0)
                - 'rho': Fluid density (default: 1.0)
                - 'A': Tank cross-sectional area (default: 1.0)
                - 'beta': Outlet coefficient (default: 1.0)
                - 'hSetpoint': Height setpoint (default: 5.0)
        """
        if parameters is None:
            parameters = {}
        
        # Set default values
        self.parameters = {
            'm': parameters.get('m', 10.0),
            'vi': parameters.get('vi', 0.0),
            'rho': parameters.get('rho', 1.0),
            'A': parameters.get('A', 1.0),
            'beta': parameters.get('beta', 1.0),
            'hSetpoint': parameters.get('hSetpoint', 5.0)
        }
        
        # Calculate and store initial derived values
        self.parameters['h'] = self.parameters['m'] / (self.parameters['rho'] * self.parameters['A'])
        self.parameters['mi'] = self.parameters['vi'] * self.parameters['rho']
        self.parameters['mo'] = self.parameters['beta'] * np.sqrt(max(self.parameters['h'], 0))
        self.parameters['v'] = self.parameters['mo'] / self.parameters['rho']
        
        # Set initial conditions
        self.initial_conditions = np.array([self.parameters['m']])
    
    def step(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Compute tank dynamics: dm/dt
        
        Args:
            X: State vector [m]
            t: Current time (unused)
            
        Returns:
            dX/dt: State derivative [dm/dt]
        """
        m = X[0]
        
        # Update current state in parameters
        self.parameters['m'] = m
        
        # Extract parameters
        vi = self.parameters['vi']
        rho = self.parameters['rho']
        A = self.parameters['A']
        beta = self.parameters['beta']
        
        # Mass balance calculations
        mi = vi * rho
        h = m / (rho * A)
        h = max(h, 0)
        mo = beta * np.sqrt(h)
        v = mo / rho
        
        # Update additional parameters for monitoring
        self.parameters['mi'] = mi
        self.parameters['h'] = h
        self.parameters['mo'] = mo
        self.parameters['v'] = v
        
        # Calculate mass derivative
        dm_dt = mi - mo
        
        return np.array([dm_dt])
