"""
author:Bruce Zhao
date: 2025/7/4 20:32
"""
import numpy as np

class Config:
    # Environment parameters
    ENV_NAME = "highway-v0"
    MAX_EPISODE_STEPS = 300

    # TD3 parameters
    BUFFER_SIZE = 100000
    BATCH_SIZE = 256
    TAU = 0.005
    GAMMA = 0.99
    POLICY_DELAY = 2

    # TSCF parameters
    SLIDING_WINDOW_SIZE = 10
    TSCF_CLUSTERS = 5
    GRANGER_LAG = 3
    MIN_EPISODES_FOR_CAUSAL = 50

    # Training parameters
    TOTAL_TIMESTEPS = 100000
    CAUSAL_UPDATE_INTERVAL = 2000

config = Config()
