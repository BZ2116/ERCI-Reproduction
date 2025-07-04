"""
author:Bruce Zhao
date: 2025/7/4 20:36
"""
import numpy as np
from train_evaluate import compare_with_baseline

if __name__ == "__main__":
    results = compare_with_baseline()
    np.savez("results.npz",
             erci_rewards=results["erci_rewards"],
             baseline_rewards=results["baseline_rewards"],
             improvement=results["improvement"])
    print("Experiment completed!")