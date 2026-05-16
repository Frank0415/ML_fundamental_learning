"""
第1章：撕开黑盒 —— 用纯 PyTorch 实现 PPO 训练 CartPole
展示 SB3 的 model.learn() 背后的核心逻辑

训练过程通过 SwanLab 记录指标（奖励曲线、损失等），
训练结束后可选弹出 GUI 窗口展示学习成果。

运行方式：
    # 默认：训练 + SwanLab 曲线（不开 GUI，速度快）
    python 2-pytorch_ppo.py

    # 打开 GUI 演示（训练完弹出小车动画窗口）
    python 2-pytorch_ppo.py --gui

关于 --gui 参数：
    训练阶段始终是 headless（无渲染），速度不受 GUI 影响。
    --gui 只控制训练结束后的演示环节是否弹出 CartPole 动画窗口。
    开启 GUI 时，演示环节每帧需要等待屏幕刷新（~16ms），会明显变慢；
    关闭 GUI 时，演示环节纯计算，几秒内跑完。
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import swanlab


# ==========================================
# 第一部分：Actor-Critic 网络（独立头 + 正交初始化）
# ==========================================
class ActorCritic(nn.Module):
    """
    独立 Actor-Critic 网络（与 SB3 MlpPolicy 对齐）：
    - Actor 和 Critic 使用各自的隐藏层，避免梯度冲突
    - 正交初始化：actor 输出层 gain=0.01 保证初始策略接近均匀分布
    """

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """正交初始化，与 SB3 默认一致"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # actor 输出层用小 gain → 初始策略接近均匀
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # critic 输出层 gain=1
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ==========================================
# 第二部分：收集轨迹（Rollout）
# ==========================================
def collect_rollout(model, env, num_steps=2048):
    """
    收集轨迹，正确处理 terminated vs truncated：
    - terminated（杆子倒了）：V(s')=0
    - truncated（达到步数上限）：V(s')需要 bootstrap
    - rollout 末尾未结束：需要 bootstrap
    """
    obs, _ = env.reset()
    transitions = []

    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        # truncated 但没 terminated → 需要存 next_obs 用于 bootstrap
        transitions.append({
            "obs": obs,
            "action": action.item(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "next_obs": next_obs if truncated and not terminated else None,
        })

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # rollout 末尾 bootstrap：如果最后一局没结束，计算 V(s_last)
    if not (terminated or truncated):
        with torch.no_grad():
            _, _, bootstrap_value = model.get_action(torch.FloatTensor(obs))
        last_bootstrap = bootstrap_value.item()
    else:
        last_bootstrap = 0.0

    return transitions, last_bootstrap


# ==========================================
# 第三部分：计算 GAE 优势
# ==========================================
def compute_gae(model, transitions, last_bootstrap, gamma=0.99, lam=0.95):
    """
    广义优势估计，正确处理：
    - terminated（真正结束）：不传播 GAE，V(s')=0
    - truncated（时间截断）：不传播 GAE，但用 V(next_obs) 作为 bootstrap
    - 正常步：正常传播 GAE
    """
    n = len(transitions)
    rewards = [t["reward"] for t in transitions]
    values = [t["value"] for t in transitions]

    # 预计算每个 truncated 步的 bootstrap value
    bootstrap_values = [0.0] * n
    for i, t in enumerate(transitions):
        if t["truncated"] and not t["terminated"] and t["next_obs"] is not None:
            with torch.no_grad():
                _, _, bv = model.get_action(torch.FloatTensor(t["next_obs"]))
            bootstrap_values[i] = bv.item()

    advantages = []
    gae = 0
    next_value = last_bootstrap

    for step in reversed(range(n)):
        t = transitions[step]

        if t["terminated"]:
            # 真正结束：V(s') = 0
            delta = rewards[step] - values[step]
            gae = delta
        elif t["truncated"]:
            # 时间截断：用 V(next_obs) bootstrap，但不传播 GAE
            delta = rewards[step] + gamma * bootstrap_values[step] - values[step]
            gae = delta
        else:
            # 正常步
            delta = rewards[step] + gamma * next_value - values[step]
            gae = delta + gamma * lam * gae

        next_value = values[step]
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ==========================================
# 第四部分：PPO 更新
# ==========================================
def ppo_update(model, optimizer, transitions, advantages, returns,
               clip_eps=0.2, epochs=10, batch_size=64):
    """PPO 裁剪目标函数更新"""
    obs = np.array([t["obs"] for t in transitions])
    actions = np.array([t["action"] for t in transitions])
    old_log_probs = np.array([t["log_prob"] for t in transitions])

    obs = torch.FloatTensor(obs)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    total_clip_frac = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))

        for start in range(0, len(transitions), batch_size):
            idx = indices[start:start + batch_size]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            # PPO 裁剪目标
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            value_loss = ((values - batch_returns) ** 2).mean()

            # 熵奖励（鼓励探索）
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.0 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # 统计指标
            with torch.no_grad():
                total_kl += (batch_old_log_probs - new_log_probs).mean().item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }


# ==========================================
# 第五部分：训练循环
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="纯 PyTorch PPO CartPole 训练")
    parser.add_argument(
        "--gui", action="store_true",
        help="训练结束后弹出 GUI 窗口演示智能体（默认关闭，仅输出得分）",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    env = gym.make("CartPole-v1")

    # 打印环境信息（状态空间、动作空间、边界阈值）
    print("=" * 50)
    print("CartPole-v1 环境信息")
    print("=" * 50)
    print(f"  观测空间:  {env.observation_space}")
    print(f"  动作空间:  {env.action_space}")
    print(f"  观测上限:  {env.observation_space.high}")
    print(f"  观测下限:  {env.observation_space.low}")
    print(f"  终止条件:  位置 > ±{env.unwrapped.x_threshold}, "
          f"角度 > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
          f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)")
    print("=" * 50)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    total_iterations = 40
    steps_per_rollout = 2048

    # 初始化 SwanLab
    swanlab.init(
        project="cartpole-pytorch",
        experiment_name="PPO-PyTorch-CartPole-v1",
        mode="local",
        config={
            "algorithm": "PPO",
            "lr": 3e-4,
            "total_iterations": total_iterations,
            "steps_per_rollout": steps_per_rollout,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "epochs": 10,
            "batch_size": 64,
        },
    )

    print("开始训练（纯 PyTorch PPO + SwanLab）...")
    print("-" * 60)

    total_timesteps = 0

    for iteration in range(total_iterations):
        # 收集数据
        transitions, last_bootstrap = collect_rollout(model, env, steps_per_rollout)

        total_timesteps += len(transitions)

        # 计算回合奖励和长度
        ep_rewards = []
        ep_lengths = []
        ep_reward = 0
        ep_length = 0
        for t in transitions:
            ep_reward += t["reward"]
            ep_length += 1
            if t["terminated"] or t["truncated"]:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward = 0
                ep_length = 0

        # 计算优势
        advantages, returns = compute_gae(model, transitions, last_bootstrap)

        # PPO 更新
        metrics = ppo_update(
            model, optimizer, transitions, advantages, returns
        )

        # 解释方差（用更新后的 Critic 重新预测，与 SB3 一致）
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(np.array([t["obs"] for t in transitions]))
            _, updated_values = model(obs_tensor)
        return_values = returns.numpy()
        updated_values_np = updated_values.numpy()
        var_returns = np.var(return_values)
        if var_returns < 1e-6:
            # 所有回报相同（如全部 500 分），EV 无意义，置为 0
            explained_variance = 0.0
        else:
            explained_variance = 1 - np.var(return_values - updated_values_np) / var_returns

        mean_reward = np.mean(ep_rewards) if ep_rewards else 0
        mean_ep_len = np.mean(ep_lengths) if ep_lengths else 0

        # 学习率线性衰减（与 SB3 默认行为一致）
        frac = 1.0 - iteration / total_iterations
        lr = 3e-4 * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 记录到 SwanLab（与 SB3 指标对齐）
        swanlab.log({
            "rollout/ep_rew_mean": mean_reward,
            "rollout/ep_len_mean": mean_ep_len,
            "train/policy_gradient_loss": metrics["policy_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/entropy_loss": -metrics["entropy"],
            "train/approx_kl": metrics["approx_kl"],
            "train/clip_fraction": metrics["clip_fraction"],
            "train/clip_range": 0.2,
            "train/explained_variance": explained_variance,
            "train/learning_rate": lr,
            "train/n_updates": (iteration + 1) * 10 * (steps_per_rollout // 64),
            "time/total_timesteps": total_timesteps,
            "time/iterations": iteration + 1,
        }, step=iteration)

        print(
            f"  迭代 {iteration + 1:2d}/{total_iterations} | "
            f"回合数: {len(ep_rewards):3d} | "
            f"平均奖励: {mean_reward:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"clip%: {metrics['clip_fraction']:.1%}"
        )

    print("-" * 60)

    # 最终评估
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action, _, _ = model.get_action(obs_tensor, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            score += reward
        eval_rewards.append(score)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\n训练完成！20 回合评估: {mean_reward:.1f} +/- {std_reward:.1f}")

    swanlab.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
    })

    # 保存模型
    torch.save(model.state_dict(), "output/pytorch_ppo_cartpole.pth")
    print(f"模型已保存到 output/pytorch_ppo_cartpole.pth")

    # GUI 演示
    if args.gui:
        try:
            vis_env = gym.make("CartPole-v1", render_mode="human")
            print("\n正在演示学习成果（5 个回合）...")
            for ep in range(5):
                obs, _ = vis_env.reset()
                done, truncated, score = False, False, 0
                while not (done or truncated):
                    obs_tensor = torch.FloatTensor(obs)
                    with torch.no_grad():
                        action, _, _ = model.get_action(obs_tensor, deterministic=True)
                    obs, reward, done, truncated, _ = vis_env.step(action.item())
                    score += reward
                print(f"  演示回合 {ep + 1} 得分: {score}")
            vis_env.close()
            print("\nGUI 演示结束。")
        except Exception:
            print("(跳过 GUI 演示，无图形界面)")
    else:
        print("\n提示: 加 --gui 可弹出小车动画窗口查看演示效果。")

    env.close()
    swanlab.finish()

    print("SwanLab 实验看板: swanlab watch swanlog")


if __name__ == "__main__":
    train()
