"""
author:Bruce Zhao
date: 2025/7/4 20:35
"""
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from config import Config

class CausalInference:
    def __init__(self, max_lag=Config.GRANGER_LAG):
        self.max_lag = max_lag
        self.causal_strengths = None

    def compute_causal_strengths(self, tscf_matrix, rewards):
        """Compute causal strengths between TSCFs and rewards using Granger causality."""
        n_episodes, n_tscf = tscf_matrix.shape
        self.causal_strengths = np.zeros(n_tscf)
        for i in range(n_tscf):
            data = np.column_stack((rewards, tscf_matrix[:, i]))
            try:
                test_result = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
                min_p_value = min([test_result[lag][0]['ssr_ftest'][1] for lag in range(1, self.max_lag + 1)])
                self.causal_strengths[i] = 1 - min_p_value
            except:
                self.causal_strengths[i] = 0.0
        if np.sum(self.causal_strengths) > 0:
            self.causal_strengths /= np.sum(self.causal_strengths)

    def get_episode_weight(self, tscf_vector):
        """Calculate causal weight for an episode."""
        if self.causal_strengths is None:
            return 1.0
        return np.dot(tscf_vector, self.causal_strengths)