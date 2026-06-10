"""
infer_toy_flow.py — Toy Rectified Flow 推理脚本

纯 numpy 实现，不依赖 torch。

用法：
    python infer_toy_flow.py --num_steps 28 --seed 0 --dim 2 \\
        --output_dir experiments/toy_rectified_flow/results

从 2D 高斯噪声出发，沿 toy vector field 做 rectified flow ODE 积分，
到达目标分布（圆环/原点/双中心/螺旋）。输出：
- trajectory JSON：记录每步的 x 坐标
- plot PNG：使用 plot_trajectory.py 绘制轨迹图

命令行参数：
    --num_steps:   ODE 积分步数（默认 28）
    --seed:        随机种子（默认 0）
    --dim:         数据维度（默认 2，仅支持 2D 可视化）
    --num_samples: 采样点数（默认 500）
    --target_type: 目标分布类型：ring|origin|dual_center|spiral（默认 ring）
    --output_dir:  输出目录（默认 experiments/toy_rectified_flow/results）
"""

import argparse
import json
import os
import sys

import numpy as np

# 确保 diffusion_engine 在 Python path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from diffusion_engine.core.rectified_flow import rectified_flow_sample
from experiments.toy_rectified_flow.train_or_load_toy_vector_field import load_toy_vector_field


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toy Rectified Flow 推理 — 从高斯噪声沿向量场 ODE 积分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认配置（ring, 28 步, seed=0）
  python infer_toy_flow.py

  # 8 步快速测试
  python infer_toy_flow.py --num_steps 8 --seed 0

  # 螺旋场，更多点
  python infer_toy_flow.py --target_type spiral --num_samples 1000
        """,
    )
    parser.add_argument(
        "--num_steps", type=int, default=28,
        help="ODE 积分步数（默认 28）"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="随机种子（默认 0）"
    )
    parser.add_argument(
        "--dim", type=int, default=2,
        help="数据维度（默认 2）"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="初始噪声点数（默认 500）"
    )
    parser.add_argument(
        "--target_type", type=str, default="ring",
        choices=["ring", "origin", "dual_center", "spiral"],
        help="目标分布类型（默认 ring）"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="输出目录（默认 experiments/toy_rectified_flow/results）"
    )
    parser.add_argument(
        "--record_every", type=int, default=1,
        help="每多少步记录一次轨迹（默认 1，每步都记录）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置确定性随机种子
    np.random.seed(args.seed)

    print(f"=== Toy Rectified Flow 推理 ===")
    print(f"  target_type : {args.target_type}")
    print(f"  num_steps   : {args.num_steps}")
    print(f"  seed        : {args.seed}")
    print(f"  dim         : {args.dim}")
    print(f"  num_samples : {args.num_samples}")
    print(f"  output_dir  : {args.output_dir}")
    print()

    # 生成初始噪声 x_T（t=1，标准高斯）
    x_T = np.random.randn(args.num_samples, args.dim).astype(np.float64)
    print(f"[1] 初始噪声生成完成，shape={x_T.shape}, "
          f"mean={x_T.mean():.4f}, std={x_T.std():.4f}")

    # 加载 toy vector field
    v_fn = load_toy_vector_field(args.target_type)

    # 构建 timestep 序列（t=1 → t=0，num_steps+1 个点）
    timesteps = np.linspace(1.0, 0.0, args.num_steps + 1, dtype=np.float64)
    print(f"[2] timesteps: {timesteps[0]:.4f} → {timesteps[-1]:.4f} "
          f"({len(timesteps)} points, {args.num_steps} steps)")

    # 轨迹记录
    trajectory = {
        "config": {
            "target_type": args.target_type,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "dim": args.dim,
            "num_samples": args.num_samples,
        },
        "initial": x_T.tolist(),
        "steps": [],
        "final": None,
    }

    def callback(i, t, t_next, x_t):
        if i % args.record_every == 0 or i == args.num_steps - 1:
            trajectory["steps"].append({
                "step": i,
                "t": float(t),
                "t_next": float(t_next),
                "points": x_t.tolist(),
            })
            if i % max(1, args.num_steps // 4) == 0 or i == args.num_steps - 1:
                # 打印进度
                r = np.sqrt(np.sum(x_t ** 2, axis=1))
                print(f"  step {i:3d}/{args.num_steps} | t={float(t):.4f} → {float(t_next):.4f} "
                      f"| mean_r={r.mean():.4f} std_r={r.std():.4f}")

    # 执行 ODE 积分
    print(f"[3] 开始 ODE 积分...")
    x_0 = rectified_flow_sample(v_fn, x_T, timesteps, callback=callback, seed=args.seed)

    trajectory["final"] = x_0.tolist()

    # 输出统计
    final_r = np.sqrt(np.sum(x_0 ** 2, axis=1))
    print(f"[4] 积分完成。")
    print(f"  final mean_r = {final_r.mean():.4f}")
    print(f"  final std_r  = {final_r.std():.4f}")
    print(f"  final min_r  = {final_r.min():.4f}")
    print(f"  final max_r  = {final_r.max():.4f}")

    # 保存 trajectory JSON
    traj_path = os.path.join(args.output_dir, f"trajectory_{args.target_type}_s{args.num_steps}_seed{args.seed}.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[5] 轨迹 JSON 已保存到: {traj_path}")

    # 尝试绘制（如果 matplotlib 可用）
    try:
        import experiments.toy_rectified_flow.plot_trajectory as plot_mod
        png_path = os.path.join(
            args.output_dir,
            f"toy_flow_{args.target_type}_s{args.num_steps}_seed{args.seed}.png"
        )
        plot_mod.plot_from_json(traj_path, png_path)
        print(f"[6] 轨迹图已保存到: {png_path}")
    except ImportError:
        print("[6] WARNING: matplotlib 不可用，跳过绘图。")
    except Exception as e:
        print(f"[6] WARNING: 绘图失败: {e}")

    # 保存结果摘要
    summary_path = os.path.join(args.output_dir, "results_summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# Toy Rectified Flow 实验结果\n\n")
        f.write(f"- **目标分布**：{args.target_type}\n")
        f.write(f"- **步数**：{args.num_steps}\n")
        f.write(f"- **种子**：{args.seed}\n")
        f.write(f"- **样本数**：{args.num_samples}\n")
        f.write(f"- **维度**：{args.dim}\n")
        f.write(f"- **初始半径**：mean={np.sqrt(np.sum(x_T**2, axis=1)).mean():.3f}\n")
        f.write(f"- **最终半径**：mean={final_r.mean():.3f}, std={final_r.std():.3f}\n")
        f.write(f"- **轨迹图**：`{os.path.basename(png_path) if 'png_path' in dir() else 'N/A'}`\n")
        f.write(f"- **轨迹数据**：`{os.path.basename(traj_path)}`\n")
    print(f"[7] 结果摘要已保存到: {summary_path}")

    print("\n=== 完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
