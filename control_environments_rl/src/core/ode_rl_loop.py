"""
Standard reinforcement learning loop for ODE environments.

This module provides a reusable RL loop implementation that follows the standard
pattern: reset, step, accumulate rewards, until done.
"""
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


def run_rl_episode(
    env, 
    agent_function: Callable,
    max_steps: Optional[int] = None,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[float], float, Dict[str, Any]]:
    """
    Run a single RL episode following the standard loop pattern.
    
    Args:
        env: ODE environment with reset() and step() methods
        agent_function: Function that takes observation and returns action
        max_steps: Maximum steps to run (default: use env.max_steps)
        verbose: Whether to print observations during episode
        
    Returns:
        observations: List of observations throughout episode
        rewards: List of rewards throughout episode  
        total_reward: Sum of all rewards
        info: Dictionary with episode statistics
    """
    # Reset environment
    observation = env.reset()
    
    # Storage
    observations = [observation.copy()]
    rewards = []
    total_reward = 0.0
    steps = 0
    
    # Determine max steps
    episode_max_steps = max_steps if max_steps is not None else env.max_steps
    
    # Main RL loop
    done = False
    while not done:
        # Agent takes action
        action = agent_function(observation)
        
        # Environment step
        observation, reward, done, step_info = env.step(action)
        
        # Store results
        observations.append(observation.copy())
        rewards.append(reward)
        total_reward += reward
        steps += 1
        
        # Optional verbose output
        if verbose:
            print(f"Step {steps}: obs={observation}, reward={reward:.3f}")
        
        # Check max steps limit
        if steps >= episode_max_steps:
            done = True
    
    # Episode summary
    info = {
        'steps': steps,
        'total_reward': total_reward,
        'final_observation': observation.copy(),
        'episode_length': steps
    }
    
    return observations, rewards, total_reward, info
