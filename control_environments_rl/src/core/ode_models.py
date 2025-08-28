"""
ODE model implementations for control environments.

This module provides built-in ODE models that can be used with the RL environment framework.
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


class HeatedTank:
    """
    Heated tank system from references.
    
    A tank with heating jacket where:
    - Tv: Tank temperature
    - Tj: Jacket temperature
    
    Heat transfer occurs between tank and jacket.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the heated tank system.
        
        Args:
            parameters: Dictionary with keys:
                - 'Tv': Initial tank temperature (default: 350.0)
                - 'Tj': Initial jacket temperature (default: 300.0)
                - 'F': Flow rate (default: 5.0)
                - 'Fj': Jacket flow rate (default: 2.0)
                - 'T0': Feed temperature (default: 300.0)
                - 'Tj0': Jacket feed temperature (default: 350.0)
                - 'V': Tank volume (default: 100.0)
                - 'Vj': Jacket volume (default: 10.0)
                - 'rho': Density (default: 1000.0)
                - 'rhoj': Jacket density (default: 1000.0)
                - 'cp': Heat capacity (default: 4.18)
                - 'cpj': Jacket heat capacity (default: 4.18)
                - 'U': Heat transfer coefficient (default: 500.0)
                - 'A': Heat transfer area (default: 2.0)
        """
        if parameters is None:
            parameters = {}
        
        # Set default values
        self.parameters = {
            'Tv': parameters.get('Tv', 350.0),
            'Tj': parameters.get('Tj', 300.0),
            'F': parameters.get('F', 5.0),
            'Fj': parameters.get('Fj', 2.0),
            'T0': parameters.get('T0', 300.0),
            'Tj0': parameters.get('Tj0', 350.0),
            'V': parameters.get('V', 100.0),
            'Vj': parameters.get('Vj', 10.0),
            'rho': parameters.get('rho', 1000.0),
            'rhoj': parameters.get('rhoj', 1000.0),
            'cp': parameters.get('cp', 4.18),
            'cpj': parameters.get('cpj', 4.18),
            'U': parameters.get('U', 500.0),
            'A': parameters.get('A', 2.0)
        }
    
    def step(self, T: np.ndarray, t: float) -> np.ndarray:
        """
        Compute temperature dynamics for tank and jacket.
        
        Args:
            T: Temperature vector [Tv, Tj]
            t: Current time (unused)
            
        Returns:
            dT/dt: Temperature derivative vector
        """
        # Extract parameters
        F = self.parameters['F']
        Fj = self.parameters['Fj']
        T0 = self.parameters['T0']
        Tj0 = self.parameters['Tj0']
        V = self.parameters['V']
        Vj = self.parameters['Vj']
        rho = self.parameters['rho']
        rhoj = self.parameters['rhoj']
        cp = self.parameters['cp']
        cpj = self.parameters['cpj']
        U = self.parameters['U']
        A = self.parameters['A']
        
        # Current temperatures
        Tv = T[0]
        Tj = T[1]
        
        # Update parameters with current state
        self.parameters['Tv'] = Tv
        self.parameters['Tj'] = Tj
        
        # Compute temperature derivatives
        dTdt = np.zeros(2)
        
        # Tank temperature derivative
        dTdt[0] = (F * (T0 - Tv))/V + (U * A * (Tj - Tv))/(V * rho * cp)
        
        # Jacket temperature derivative  
        dTdt[1] = (Fj * (Tj0 - Tj))/Vj - (U * A * (Tj - Tv))/(Vj * rhoj * cpj)
        
        return dTdt
