"""
Controlled tank example with RL loop.
"""
import numpy as np
import matplotlib.pyplot as plt
from control_environments_rl import ODEEnvironment, ode_models


def height_tracking_reward(model):
    """Reward function: negative squared error from height setpoint."""
    error = model.parameters['hSetpoint'] - model.parameters['h']
    return -error**2

# Create tank and environment
tank = ode_models.ControlledTank()
env = ODEEnvironment(
    model=tank, 
    time_step=0.5,
    max_steps=1000,
    reward_function=height_tracking_reward,
    observation_variables=['m', 'h', 'hSetpoint'],
    action_variables=['vi']
)

# ============================================================================== #

# Create PID controller as agent
class SimplePID:
    """Simple PID controller for tank height control."""
    
    def __init__(self, kp=0.1, ki=0.1, kd=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        self.integral = 0.0
        self.prev_error = 0.0
    
    def __call__(self, obs, setpoint):
        """PID control action based on observation."""
        # obs contains [m, h] based on observation_variables
        current_h = obs[1]  # height is second observation
        
        # Calculate error
        error = setpoint - current_h
        
        # Update integral
        self.integral += error
        
        # Calculate derivative
        derivative = error - self.prev_error
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        # Ensure non-negative flow
        output = max(0.0, output)
        
        return np.array([output])

# ============================================================================== #

# Run episode
pid_controller = SimplePID(kp=0.1, ki=0.1, kd=0.0)
obs = env.reset()
rewards = []

for step in range(100):
    action = pid_controller(obs, env.model.parameters['hSetpoint'])  # PID control
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done: break

# Plot results
times = np.arange(len(list(env.model_parameter_history.values())[0])) * 0.1

plt.figure(figsize=(12, 3))

plt.subplot(1, 3, 1)
plt.plot(times, env.model_parameter_history['h'], 'g-', label='Height', marker='o', markersize=2)
plt.plot(times, env.model_parameter_history['hSetpoint'], 'r--', label='Setpoint', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Height')
plt.title('Tank Height vs Setpoint')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(times, env.model_parameter_history['vi'], 'b-', label='Inlet Flow', marker='o', markersize=2)
plt.xlabel('Time')
plt.ylabel('Inlet Flow')
plt.title('Control Input')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(times[1:], rewards, 'r-', marker='o', markersize=2)
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Rewards')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
