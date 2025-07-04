"""
author:Bruce Zhao
date: 2025/7/4 20:35
"""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from tscf import TSCFExtractor
from causal_inference import CausalInference
from config import Config

class ERCIReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='auto'):
        super().__init__(buffer_size, observation_space, action_space, device=device)
        self.episodes = []
        self.episode_data = []
        self.episode_weights = np.zeros(buffer_size)
        self.episode_tscf = {}
        self.tscf_extractor = TSCFExtractor()
        self.causal_inference = CausalInference()
        self.causal_update_counter = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        """Add new transition to buffer."""
        super().add(obs, next_obs, action, reward, done, infos)
        self.episode_data.append((obs, action, reward, done))
        if done:
            self.episodes.append(self.episode_data)
            self.episode_data = []
            if len(self.episodes) >= Config.MIN_EPISODES_FOR_CAUSAL:
                self.causal_update_counter += 1
                if self.causal_update_counter % Config.CAUSAL_UPDATE_INTERVAL == 0:
                    self.update_causal_model()

    def update_causal_model(self):
        """Update causal model using collected episodes."""
        all_sequences = []
        episode_rewards = []
        for episode in self.episodes:
            total_reward = sum([step[2] for step in episode])
            episode_rewards.append(total_reward)
            sequences = self.tscf_extractor.extract_features(episode)
            all_sequences.extend(sequences)
        if len(all_sequences) > 0:
            self.tscf_extractor.fit(all_sequences)
            tscf_matrix = []
            for episode in self.episodes:
                sequences = self.tscf_extractor.extract_features(episode)
                tscf_vector = self.tscf_extractor.transform(sequences)
                tscf_matrix.append(tscf_vector)
            tscf_matrix = np.array(tscf_matrix)
            episode_rewards = np.array(episode_rewards)
            self.causal_inference.compute_causal_strengths(tscf_matrix, episode_rewards)
            for i, episode in enumerate(self.episodes):
                weight = self.causal_inference.get_episode_weight(tscf_matrix[i])
                start_idx = self.pos - len(episode) if self.pos >= len(episode) else 0
                end_idx = self.pos
                for j in range(start_idx, end_idx):
                    self.episode_weights[j] = weight

    def sample(self, batch_size: int):
        """Sample transitions using causal weights."""
        weights = self.episode_weights[:self.pos]
        probs = None if np.sum(weights) == 0 else weights / np.sum(weights)
        indices = np.random.choice(self.pos, size=batch_size, p=probs)
        return self._get_samples(indices)