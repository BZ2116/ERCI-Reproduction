"""
author:Bruce Zhao
date: 2025/7/4 20:36
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from model import ERCITD3
from environment import make_env
from config import Config

def train_and_evaluate():
    config = Config()
    env = make_env()
    eval_env = make_env()
    model = ERCITD3(
        "MlpPolicy",
        env,
        buffer_size=config.BUFFER_SIZE,
        learning_starts=1000,
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        policy_delay=config.POLICY_DELAY,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    eval_results = []

    def eval_callback(locals_, globals_):
        nonlocal eval_results
        if locals_['self'].num_timesteps % 5000 == 0:
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
            eval_results.append((locals_['self'].num_timesteps, mean_reward))
            print(f"Timesteps: {locals_['self'].num_timesteps}, Eval Reward: {mean_reward:.2f}")
        return True

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=eval_callback,
        log_interval=4
    )
    model.save("erci_td3_highway")
    timesteps, rewards = zip(*eval_results)
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.title("ERCI-TD3 Training Performance")
    plt.grid(True)
    plt.savefig("training_curve.png")
    plt.close()
    return model, eval_results

def compare_with_baseline():
    config = Config()
    env = make_env()
    erci_model, erci_results = train_and_evaluate()
    baseline_model = TD3(
        "MlpPolicy",
        env,
        buffer_size=config.BUFFER_SIZE,
        learning_starts=1000,
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        policy_delay=config.POLICY_DELAY,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    baseline_results = []

    def baseline_callback(locals_, globals_):
        if locals_['self'].num_timesteps % 5000 == 0:
            mean_reward, _ = evaluate_policy(baseline_model, env, n_eval_episodes=5)
            baseline_results.append((locals_['self'].num_timesteps, mean_reward))
            print(f"Timesteps: {locals_['self'].num_timesteps}, Baseline Reward: {mean_reward:.2f}")
        return True

    baseline_model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=baseline_callback,
        log_interval=4
    )
    baseline_model.save("td3_highway")
    plt.figure(figsize=(12, 8))
    erci_timesteps, erci_rewards = zip(*erci_results)
    plt.plot(erci_timesteps, erci_rewards, label='ERCI-TD3', marker='o')
    baseline_timesteps, baseline_rewards = zip(*baseline_results)
    plt.plot(baseline_timesteps, baseline_rewards, label='TD3 Baseline', marker='s')
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.title("ERCI vs Baseline TD3 Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison.png")
    plt.close()
    final_erci_reward = erci_rewards[-1]
    final_baseline_reward = baseline_rewards[-1]
    improvement = (final_erci_reward - final_baseline_reward) / abs(final_baseline_reward) * 100
    print(f"ERCI Performance Improvement: {improvement:.2f}%")
    return {
        "erci_rewards": erci_rewards,
        "baseline_rewards": baseline_rewards,
        "improvement": improvement
    }