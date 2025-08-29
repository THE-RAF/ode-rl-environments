"""
ODE-RL Environments - Transform ODEs into RL environments.

This package provides tools for converting ordinary differential equation (ODE) 
systems into reinforcement learning environments, making it easy to apply RL 
techniques to control problems.

Main Components:
- ODEEnvironment: Core RL environment for ODE systems
- ode_models: Package containing pre-built ODE models
"""

from .src.core.ode_environment import ODEEnvironment
from .src import ode_models

__version__ = "1.0.0"
__all__ = ['ODEEnvironment', 'ode_models']
