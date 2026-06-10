"""
train_or_load_toy_vector_field.py — Toy Vector Field 定义（无需训练）

纯 numpy 实现，不依赖 torch。

提供多种合成向量场，用于演示 rectified flow ODE 的行为：
- ring: 将所有点推送到半径 R 的圆环上
- origin: 将所有点拉向原点
- dual_center: 双中心吸引（演示流形学习）
- spiral: 将噪声散点沿螺旋线收紧

设计原则：
    每个 vector field 函数签名为 v(x, t)，返回与 x 同 shape 的向量。
    t ∈ [0,1]，t=1 为噪声端（力较弱），t=0 为数据端（力较强）。
    这模拟了 rectified flow 中"靠近数据时约束更强"的行为。
"""

import numpy as np


def _as_float(t):
    """安全地将 t 转为标量 float（处理 numpy scalar、list 等）。"""
    if isinstance(t, np.ndarray):
        return float(t.flat[0])
    return float(t)


def toy_vector_field_ring(x, t, radius=3.0):
    """
    环形吸引子：将所有点推送到半径为 radius 的圆环上。

    力的大小随 t 减小而增强：在 t=1 时力 ≈ 0（允许初始扩散），
    在 t→0 时力最大（强制收敛到圆环）。

    向量场方向：radial 方向，指向圆环（若点在内侧则向外，外侧则向内）。
    """
    t_val = _as_float(t)
    r = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
    r_safe = np.maximum(r, 1e-8)

    # 投影到圆环上的目标点
    target = radius * x / r_safe
    # 力：从当前点到目标点。t=0 时力最大，t=1 时力为 0。
    strength = 0.5 * (1.0 - np.cos(np.pi * (1.0 - t_val)))
    return strength * (target - x)


def toy_vector_field_origin(x, t):
    """
    原点吸引子：所有点流向原点。

    这是最简单的"去噪"演示：从任意高斯散点回到单一中心。
    等价于 v = -x，但添加了 t 依赖的强度调制（t→0 时收敛更快）。
    """
    t_val = _as_float(t)
    # t 越小，力越大（>1 的系数）
    strength = 1.0 / (t_val + 0.1)
    return -strength * x


def toy_vector_field_dual_center(x, t, center_left=(-2.0, 0.0), center_right=(2.0, 0.0)):
    """
    双中心吸引子：根据 x 的横坐标选择吸引中心。

    x > 0 → 流向右中心 (+2, 0)
    x ≤ 0 → 流向左中心 (-2, 0)

    演示 rectified flow 可以将一个分布一分为二。
    """
    t_val = _as_float(t)
    center_left = np.array(center_left, dtype=x.dtype)
    center_right = np.array(center_right, dtype=x.dtype)

    # 按第一维（x 轴）选择中心
    mask_right = (x[..., 0:1] > 0).astype(x.dtype)
    center = mask_right * center_right + (1 - mask_right) * center_left

    strength = 1.0 / (t_val + 0.15)
    return strength * (center - x)


def toy_vector_field_spiral(x, t):
    """
    螺旋收紧场：径向向原点 + 切向旋转，演示非梯度场流动。

    不满足梯度场的旋度为零条件（∂v_y/∂x ≠ ∂v_x/∂y），
    演示 rectified flow 可以学习非保守向量场。
    """
    t_val = _as_float(t)
    r = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
    r_safe = np.maximum(r, 1e-8)

    # 径向分量：指向原点
    radial = -x / r_safe
    # 切向分量：顺时针旋转
    tangential = np.stack([x[..., 1:2], -x[..., 0:1]], axis=-1)

    strength_radial = 0.5 / (t_val + 0.1)
    strength_tang = 0.3 * (1.0 - t_val)

    return strength_radial * radial + strength_tang * tangential


# 注册表
TOY_VECTOR_FIELDS = {
    "ring": toy_vector_field_ring,
    "origin": toy_vector_field_origin,
    "dual_center": toy_vector_field_dual_center,
    "spiral": toy_vector_field_spiral,
}


def load_toy_vector_field(target_type="ring", **kwargs):
    """
    加载 toy vector field（无训练，直接返回合成函数）。

    参数：
        target_type: "ring" | "origin" | "dual_center" | "spiral"
        **kwargs: 传递给具体 vector field 的额外参数（如 radius）。

    返回：
        v_fn: 可调用对象 v_fn(x, t) → vector_field
    """
    if target_type not in TOY_VECTOR_FIELDS:
        raise ValueError(
            f"未知 target_type: {target_type}。可选: {list(TOY_VECTOR_FIELDS.keys())}"
        )
    base_fn = TOY_VECTOR_FIELDS[target_type]

    def v_fn(x, t):
        return base_fn(x, t, **kwargs)

    return v_fn


if __name__ == "__main__":
    # 快速 smoke test
    print("=== Toy Vector Field Smoke Test ===")
    for name in TOY_VECTOR_FIELDS:
        v_fn = load_toy_vector_field(name)
        x = np.random.randn(5, 2).astype(np.float64)
        for t_val in [1.0, 0.5, 0.0]:
            v = v_fn(x, t_val)
            assert v.shape == (5, 2), f"{name}: shape mismatch {v.shape}"
        print(f"  {name}: OK (shapes correct)")
    print("All vector fields passed smoke test.")
