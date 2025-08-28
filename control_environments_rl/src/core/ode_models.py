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


class ChemicalReactor:
    """
    Chemical reactor system with reaction A + 2B → 3C.
    
    A continuous stirred-tank reactor (CSTR) with three components:
    - A, B: Reactants
    - C: Product
    
    The system tracks molar amounts (Na, Nb, Nc) which are the state variables.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the chemical reactor system.
        
        Args:
            parameters: Dictionary with keys:
                - 'Na': Initial moles of A (default: 0.0)
                - 'Nb': Initial moles of B (default: 0.0)
                - 'Nc': Initial moles of C (default: 0.0)
                - 'vai': Volumetric flow rate of stream A (default: 0.0)
                - 'vbi': Volumetric flow rate of stream B (default: 0.0)
                - 'vci': Volumetric flow rate of stream C (default: 0.0)
                - 'V': Reactor volume (default: 1.0)
                - 'Cai': Inlet concentration of A (default: 1.0)
                - 'Cbi': Inlet concentration of B (default: 1.0)
                - 'Cci': Inlet concentration of C (default: 1.0)
                - 'k': Reaction rate constant (default: 100.0)
        """
        if parameters is None:
            parameters = {}
        
        # Set default values
        self.parameters = {
            'Na': parameters.get('Na', 0.0),
            'Nb': parameters.get('Nb', 0.0),
            'Nc': parameters.get('Nc', 0.0),
            'vai': parameters.get('vai', 0.0),
            'vbi': parameters.get('vbi', 0.0),
            'vci': parameters.get('vci', 0.0),
            'V': parameters.get('V', 1.0),
            'Cai': parameters.get('Cai', 1.0),
            'Cbi': parameters.get('Cbi', 1.0),
            'Cci': parameters.get('Cci', 1.0),
            'k': parameters.get('k', 100.0)
        }
        
        # Set initial conditions
        self.initial_conditions = np.array([self.parameters['Na'], self.parameters['Nb'], self.parameters['Nc']])
    
    def step(self, N: np.ndarray, t: float) -> np.ndarray:
        """
        Compute reactor dynamics: dN/dt for molar amounts.
        
        Args:
            N: Molar amounts vector [Na, Nb, Nc]
            t: Current time (unused)
            
        Returns:
            dN/dt: Molar amount derivatives [dNa/dt, dNb/dt, dNc/dt]
        """
        # Current molar amounts
        Na, Nb, Nc = N[0], N[1], N[2]
        
        # Update parameters with current state
        self.parameters['Na'] = Na
        self.parameters['Nb'] = Nb
        self.parameters['Nc'] = Nc
        
        # Extract parameters
        vai = self.parameters['vai']
        vbi = self.parameters['vbi']
        vci = self.parameters['vci']
        V = self.parameters['V']
        Cai = self.parameters['Cai']
        Cbi = self.parameters['Cbi']
        Cci = self.parameters['Cci']
        k = self.parameters['k']
        
        # Calculate total volumetric flow rate
        v = vai + vbi + vci
        
        # Calculate reactor concentrations
        Ca = Na / V if V > 0 else 0.0
        Cb = Nb / V if V > 0 else 0.0
        Cc = Nc / V if V > 0 else 0.0
        
        # Calculate inlet molar flow rates
        Fai = vai * Cai
        Fbi = vbi * Cbi
        Fci = vci * Cci
        
        # Calculate outlet molar flow rates
        Fa = v * Ca
        Fb = v * Cb
        Fc = v * Cc
        
        # Calculate reaction rate: A + 2B -> 3C
        # r = k * Ca * Cb^2
        r = k * Ca * (Cb ** 2)
        
        # Calculate individual species reaction rates
        ra = -r      # A is consumed (stoichiometry: -1)
        rb = -2 * r  # B is consumed (stoichiometry: -2) 
        rc = 3 * r   # C is produced (stoichiometry: +3)
        
        # Update additional parameters for monitoring
        self.parameters['v'] = v
        self.parameters['Ca'] = Ca
        self.parameters['Cb'] = Cb
        self.parameters['Cc'] = Cc
        self.parameters['r'] = r
        self.parameters['ra'] = ra
        self.parameters['rb'] = rb
        self.parameters['rc'] = rc
        
        # Calculate molar amount derivatives
        dNdt = np.zeros(3)
        
        # dNa/dt = Fai - Fa + ra * V
        dNdt[0] = Fai - Fa + ra * V
        
        # dNb/dt = Fbi - Fb + rb * V  
        dNdt[1] = Fbi - Fb + rb * V
        
        # dNc/dt = Fci - Fc + rc * V
        dNdt[2] = Fci - Fc + rc * V
        
        return dNdt
