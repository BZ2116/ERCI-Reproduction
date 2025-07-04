"""
author:Bruce Zhao
date: 2025/7/4 20:36
"""
from stable_baselines3 import TD3
from replay_buffer import ERCIReplayBuffer
from config import Config

class ERCITD3(TD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_replay_buffer(self):
        """Create ERCI replay buffer."""
        return ERCIReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
        )