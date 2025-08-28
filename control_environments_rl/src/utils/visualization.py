"""
Visualization utilities for control environments.

This module provides plotting functions for system states, rewards, and trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any


def plot_episode_results(
    states: List[np.ndarray],
    rewards: List[float],
    times: List[float],
    actions: Optional[List[np.ndarray]] = None,
    state_labels: Optional[List[str]] = None,
    title: str = "Episode Results"
) -> None:
    """
    Plot episode results showing states, rewards, and optionally actions over time.
    
    Args:
        states: List of state vectors over time
        rewards: List of rewards over time
        times: List of time values
        actions: Optional list of action vectors over time
        state_labels: Optional labels for state variables (e.g., ['x', 'y'])
        title: Plot title
    """
    n_states = len(states[0]) if states else 0
    n_plots = 2 + (1 if actions is not None else 0)  # States + Rewards + Actions (optional)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Plot states
    ax_states = axes[0]
    states_array = np.array(states)
    
    if state_labels is None:
        state_labels = [f'State {i}' for i in range(n_states)]
    
    for i in range(n_states):
        ax_states.plot(times, states_array[:, i], label=state_labels[i], linewidth=2)
    
    ax_states.set_xlabel('Time')
    ax_states.set_ylabel('State Values')
    ax_states.set_title(f'{title} - System States')
    ax_states.legend()
    ax_states.grid(True, alpha=0.3)
    
    # Plot rewards
    ax_rewards = axes[1]
    ax_rewards.plot(times, rewards, 'r-', linewidth=2, label='Reward')
    ax_rewards.set_xlabel('Time')
    ax_rewards.set_ylabel('Reward')
    ax_rewards.set_title(f'{title} - Rewards')
    ax_rewards.legend()
    ax_rewards.grid(True, alpha=0.3)
    
    # Plot actions if provided
    if actions is not None and len(axes) > 2:
        ax_actions = axes[2]
        actions_array = np.array(actions)
        n_actions = actions_array.shape[1] if len(actions_array.shape) > 1 else 1
        
        if n_actions == 1:
            ax_actions.plot(times, actions_array, 'g-', linewidth=2, label='Action')
        else:
            for i in range(n_actions):
                ax_actions.plot(times, actions_array[:, i], label=f'Action {i}', linewidth=2)
        
        ax_actions.set_xlabel('Time')
        ax_actions.set_ylabel('Action Values')
        ax_actions.set_title(f'{title} - Actions')
        ax_actions.legend()
        ax_actions.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_phase_portrait(
    states: List[np.ndarray],
    title: str = "Phase Portrait",
    state_labels: Optional[List[str]] = None
) -> None:
    """
    Plot 2D phase portrait for 2-state systems.
    
    Args:
        states: List of 2D state vectors
        title: Plot title
        state_labels: Labels for state variables (e.g., ['x', 'y'])
    """
    if len(states[0]) != 2:
        print("Phase portrait only available for 2D systems")
        return
    
    states_array = np.array(states)
    
    if state_labels is None:
        state_labels = ['State 0', 'State 1']
    
    plt.figure(figsize=(8, 8))
    
    # Plot trajectory
    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Mark start and end points
    plt.plot(states_array[0, 0], states_array[0, 1], 'go', markersize=10, label='Start')
    plt.plot(states_array[-1, 0], states_array[-1, 1], 'ro', markersize=10, label='End')
    
    # Add arrows to show direction
    n_arrows = min(10, len(states_array) // 5)
    for i in range(0, len(states_array) - 1, len(states_array) // n_arrows):
        dx = states_array[i + 1, 0] - states_array[i, 0]
        dy = states_array[i + 1, 1] - states_array[i, 1]
        plt.arrow(states_array[i, 0], states_array[i, 1], dx, dy, 
                 head_width=0.05 * max(abs(dx), abs(dy)), 
                 head_length=0.1 * max(abs(dx), abs(dy)), 
                 fc='blue', ec='blue', alpha=0.6)
    
    plt.xlabel(state_labels[0])
    plt.ylabel(state_labels[1])
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def plot_reward_analysis(
    rewards: List[float],
    times: List[float],
    window_size: int = 10,
    title: str = "Reward Analysis"
) -> None:
    """
    Plot detailed reward analysis with moving average.
    
    Args:
        rewards: List of rewards over time
        times: List of time values
        window_size: Window size for moving average
        title: Plot title
    """
    rewards_array = np.array(rewards)
    
    # Calculate moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
        avg_times = times[window_size-1:]
    else:
        moving_avg = rewards_array
        avg_times = times
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot raw rewards
    ax1.plot(times, rewards, 'b-', alpha=0.6, linewidth=1, label='Raw Rewards')
    ax1.plot(avg_times, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'{title} - Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot reward histogram
    ax2.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.3f}')
    ax2.set_xlabel('Reward Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title} - Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nReward Statistics:")
    print(f"Mean: {np.mean(rewards):.4f}")
    print(f"Std:  {np.std(rewards):.4f}")
    print(f"Min:  {np.min(rewards):.4f}")
    print(f"Max:  {np.max(rewards):.4f}")


def plot_state_evolution(
    states: List[np.ndarray],
    times: List[float],
    state_labels: Optional[List[str]] = None,
    title: str = "State Evolution"
) -> None:
    """
    Plot individual state variables evolution over time.
    
    Args:
        states: List of state vectors over time
        times: List of time values
        state_labels: Optional labels for state variables
        title: Plot title
    """
    states_array = np.array(states)
    n_states = states_array.shape[1]
    
    if state_labels is None:
        state_labels = [f'x_{i}' for i in range(n_states)]
    
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2 * n_states))
    if n_states == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i in range(n_states):
        color = colors[i % len(colors)]
        axes[i].plot(times, states_array[:, i], color=color, linewidth=2)
        axes[i].set_ylabel(state_labels[i])
        axes[i].set_title(f'{title} - {state_labels[i]} vs Time')
        axes[i].grid(True, alpha=0.3)
        
        if i == n_states - 1:  # Last subplot
            axes[i].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
