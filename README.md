# ERCI-Reproduction
## ERCI的复现
该资源库包含论文中描述的用于深度强化学习（DRL）的ERCI（带因果推理的经验重放）方法的模块化实现：


ERCI: An Explainable Experience Replay Approach with Causal Inference for Deep Reinforcement Learning
Jingwen Wang, Dehui Du, Lili Tian, Yikang Chen, Vida Li, YiYang Li
The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)
Shanghai Key Laboratory of Trustworthy Computing, East China Normal University, 200062, China

为实现可重复性，该方法简化了最初的 ERCI 方法，重点关注多变量时间序列表示（TSCF 提取）、使用格兰杰因果关系的因果推断以及用于因果加权采样的改进重放缓冲器等关键部分。 代码使用 PyTorch、Stable-Baselines3 和 Highway-Env 环境构建。


## 输出：

1. 训练好的模型保存为 erci_td3_highway.zip 和 td3_highway.zip。
2. 训练曲线保存为 training_curve.png 和 comparison.png。
3. 结果（奖励和性能提升）保存为 results.npz。
