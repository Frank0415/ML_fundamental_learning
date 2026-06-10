"""
plot_trajectory.py — 从 trajectory JSON 绘制 Rectified Flow 轨迹图

纯 matplotlib + numpy 实现，不依赖 torch。
支持：
- 从 JSON 读取轨迹数据
- 绘制初始分布 + 中间步 + 最终分布
- 可选绘制几条样本的完整轨迹线
- 保存为 PNG

用法：
    python plot_trajectory.py trajectory.json output.png
    或作为模块被 infer_toy_flow.py 调用：
    from plot_trajectory import plot_from_json
"""

import json
import sys
import numpy as np


def plot_from_json(traj_json_path, output_png_path, num_trajectories=30):
    """
    从 trajectory JSON 文件读取数据并绘制轨迹图。

    参数：
        traj_json_path: trajectory JSON 文件路径。
        output_png_path: 输出 PNG 文件路径。
        num_trajectories: 要绘制的单独轨迹线条数（0 表示不绘制）。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无 GUI 后端
    import matplotlib.pyplot as plt

    # 尝试使用支持中文的字体
    _cjk_fonts = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
                  "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei"]
    for _font in _cjk_fonts:
        try:
            from matplotlib.font_manager import FontProperties
            FontProperties(family=_font)
            plt.rcParams["font.family"] = _font
            break
        except Exception:
            continue

    with open(traj_json_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    initial = np.array(data["initial"])
    steps = data["steps"]
    final = np.array(data["final"])

    # 选择要展示的步（首批、中间、最后 + 均匀采样）
    n_steps = len(steps)
    show_indices = [0]
    if n_steps > 3:
        mid = n_steps // 2
        show_indices.append(mid)
    if n_steps > 2:
        show_indices.append(n_steps - 1)
    show_indices = sorted(set(show_indices))

    # 若步数多，再加均匀采样
    if n_steps > 5:
        extra = list(range(1, n_steps - 1, max(1, n_steps // 4)))
        show_indices = sorted(set(show_indices + extra))
        show_indices = show_indices[:6]  # 最多 6 个子图

    n_plots = 1 + len(show_indices)  # 初始 + 展示步
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    target_type = config.get("target_type", "ring")

    # 子图 0：初始噪声
    ax = axes[0]
    ax.scatter(initial[:, 0], initial[:, 1], s=3, alpha=0.5, c="blue", label="t=1 (noise)")
    ax.set_title(f"t=1.0 (初始噪声)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 按收敛程度着色
    for plot_i, step_idx in enumerate(show_indices):
        ax = axes[plot_i + 1]
        step_data = steps[step_idx]
        pts = np.array(step_data["points"])
        t_val = step_data["t"]

        # 按半径着色
        r = np.sqrt(np.sum(pts ** 2, axis=1))
        scatter = ax.scatter(pts[:, 0], pts[:, 1], s=3, c=r, cmap="viridis", alpha=0.6)
        ax.set_title(f"step {step_data['step']:d} (t={t_val:.3f})", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, shrink=0.8, label="r")

    # 隐藏多余的子图
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Toy Rectified Flow — {target_type} "
        f"({config['num_steps']} 步, seed={config['seed']})",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 第二张图：最终分布 + 样本轨迹线
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(final[:, 0], final[:, 1], s=3, alpha=0.5, c="red", label=f"final (t=0)")

    # 绘制几条样本的完整轨迹
    if num_trajectories > 0:
        n_traj = min(num_trajectories, len(initial))
        indices = np.linspace(0, len(initial) - 1, n_traj, dtype=int)

        for idx in indices:
            traj_x = [initial[idx, 0]]
            traj_y = [initial[idx, 1]]
            for step_data in steps:
                pts = np.array(step_data["points"])
                traj_x.append(pts[idx, 0])
                traj_y.append(pts[idx, 1])

            # 从蓝（噪声）到红（数据）的颜色渐变
            ax2.plot(traj_x, traj_y, "k-", alpha=0.15, linewidth=0.5)
            ax2.scatter(traj_x[0], traj_y[0], s=10, c="blue", alpha=0.4, zorder=5)
            ax2.scatter(traj_x[-1], traj_y[-1], s=10, c="red", alpha=0.6, zorder=5)

    ax2.set_title(
        f"Toy Rectified Flow — {target_type} 最终分布 + {num_trajectories} 条轨迹",
        fontsize=12,
    )
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    traj_png_path = output_png_path.replace(".png", "_trajectories.png")
    fig2.savefig(traj_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return output_png_path


def main():
    if len(sys.argv) < 3:
        print("用法: python plot_trajectory.py <trajectory.json> <output.png> [num_trajectories]")
        print("示例: python plot_trajectory.py trajectory_ring_s28_seed0.json toy_flow_ring.png 20")
        sys.exit(1)

    json_path = sys.argv[1]
    png_path = sys.argv[2]
    n_traj = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    result = plot_from_json(json_path, png_path, num_trajectories=n_traj)
    print(f"图片已保存到: {result}")


if __name__ == "__main__":
    main()
