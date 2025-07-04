"""
author:Bruce Zhao
date: 2025/7/4 20:33
"""
import gymnasium as gym
from highway_env.envs import HighwayEnv
from gymnasium.wrappers import TimeLimit
from config import Config

def make_env():
    config = Config()
    env = HighwayEnv(config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True
        },
        "action": {
            "type": "ContinuousAction",
            "steering_range": [-np.pi / 4, np.pi / 4],
            "longitudinal": True,
            "lateral": True
        },
        "duration": config.MAX_EPISODE_STEPS,
        "collision_reward": -1,
        "high_speed_reward": 0.4,
        "lane_centering_cost": 0.1,
        "normalize": True
    })
    env = TimeLimit(env, max_episode_steps=config.MAX_EPISODE_STEPS)
    return env