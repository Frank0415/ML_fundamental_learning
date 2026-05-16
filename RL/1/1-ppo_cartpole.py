"""
第1章：使用 Stable-Baselines3 的 PPO 训练 CartPole

训练过程通过 SwanLab 记录指标（奖励曲线、损失等），
训练结束后可选弹出 GUI 窗口展示学习成果。

运行方式：
    # 默认：训练 + SwanLab 曲线（不开 GUI，速度快）
    python 1-ppo_cartpole.py

    # 打开 GUI 演示（训练完弹出小车动画窗口）
    python 1-ppo_cartpole.py --gui

关于 --gui 参数：
    训练阶段始终是 headless（无渲染），速度不受 GUI 影响。
    --gui 只控制训练结束后的演示环节是否弹出 CartPole 动画窗口。
    开启 GUI 时，演示环节每帧需要等待屏幕刷新（~16ms），会明显变慢；
    关闭 GUI 时，演示环节纯计算，几秒内跑完。
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from swanlab.integration.sb3 import SwanLabCallback
import swanlab


class LogApproxKL(BaseCallback):
    """补录 train/approx_kl 到 SwanLab。

    SB3 的 PPO.train() 内部通过 logger.record("train/approx_kl", ...) 记录了该指标，
    但值为 numpy.float32 类型。SwanLab 的 SB3 回调在 write() 中使用
    isinstance(value, (int, float)) 做类型检查，而 numpy.float32 不通过该检查
    （numpy.float64 和 Python float 可以通过），导致 approx_kl 被静默跳过。

    本回调在每次 train() 执行完毕后，从 logger 缓存中取出 approx_kl 值，
    转为 Python float 后直接通过 swanlab.log 补录。
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # train() 已在 _on_rollout_end 触发前执行完毕，
        # logger 缓存中包含本轮 train 的所有指标。
        logger = self.model.logger
        if hasattr(logger, "name_to_value") and "train/approx_kl" in logger.name_to_value:
            value = float(logger.name_to_value["train/approx_kl"])
            swanlab.log({"train/approx_kl": value}, step=self.num_timesteps)


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO CartPole 训练")
    parser.add_argument(
        "--gui", action="store_true",
        help="训练结束后弹出 GUI 窗口演示智能体（默认关闭，仅输出得分）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    # ==========================================
    # 第一阶段：训练
    # ==========================================
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

    model = PPO("MlpPolicy", env, verbose=1)

    print("开始训练（带 SwanLab 日志）...")
    swanlab_cb = SwanLabCallback(
        project="cartpole-ppo",
        experiment_name="PPO-CartPole-v1",
        mode="local",
    )
    model.learn(
        total_timesteps=80000,
        callback=[swanlab_cb, LogApproxKL()],
    )

    # 评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"训练完成！平均奖励: {mean_reward} +/- {std_reward}")

    model.save("output/ppo_cartpole")
    env.close()

    # ==========================================
    # 第二阶段：演示学习成果
    # ==========================================
    print("\n正在展示智能体的学习成果...")
    render_mode = "human" if args.gui else None
    vis_env = gym.make("CartPole-v1", render_mode=render_mode)
    model = PPO.load("output/ppo_cartpole")

    for episode in range(5):
        obs, info = vis_env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vis_env.step(action)
            score += reward
        print(f"  回合 {episode + 1} 得分: {score}")

    vis_env.close()

    if args.gui:
        print("\nGUI 演示结束。")
    else:
        print("\n提示: 加 --gui 可弹出小车动画窗口查看演示效果。")

    print("SwanLab 实验看板: swanlab watch swanlog")


if __name__ == "__main__":
    main()
