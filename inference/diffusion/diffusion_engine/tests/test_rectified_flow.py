"""
test_rectified_flow.py — rectified_flow_step 与 rectified_flow_sample 的 pytest 测试

覆盖：
- v=常数 时直线插值：x_t 在 x_T 和 x_0 预测的直线上
- v=0 时 x 保持不变
- rectified_flow_sample 完整循环收敛
- 多次调用相同 seed 的可复现性
"""

import numpy as np
import pytest
from diffusion_engine.core.rectified_flow import (
    rectified_flow_step,
    rectified_flow_sample,
)


# ═══════════════════════════════════════════════════════════════════════════
# rectified_flow_step 测试
# ═══════════════════════════════════════════════════════════════════════════

class TestRectifiedFlowStep:

    def test_constant_v_linear_motion(self):
        """v=常数时，x_t 应在直线上移动。"""
        x_t = np.array([1.0, 2.0], dtype=np.float64)
        v = np.array([3.0, -1.0], dtype=np.float64)

        result = rectified_flow_step(x_t, t=1.0, t_next=0.5, v_theta=v)

        # dt = 0.5 - 1.0 = -0.5
        expected = x_t + (-0.5) * v
        assert np.allclose(result, expected, atol=1e-10)

    def test_zero_v_no_change(self):
        """v=0 时 x 不应改变。"""
        x_t = np.array([5.0, -3.0, 2.0], dtype=np.float64)
        v = np.zeros_like(x_t)

        result = rectified_flow_step(x_t, t=1.0, t_next=0.0, v_theta=v)
        assert np.allclose(result, x_t, atol=1e-10)

    def test_single_negative_step(self):
        """单步负 dt（t 递减）应沿 v 方向移动。"""
        x_t = np.array([0.0, 0.0])
        v = np.array([2.0, 2.0])

        result = rectified_flow_step(x_t, t=1.0, t_next=0.0, v_theta=v)
        # dt = -1.0, result = [0,0] + (-1)*[2,2] = [-2,-2]
        assert np.allclose(result, np.array([-2.0, -2.0]), atol=1e-10)

    def test_single_positive_step(self):
        """单步正 dt（t 递增，反向积分）应沿 v 反方向移动。"""
        x_t = np.array([0.0, 0.0])
        v = np.array([2.0, 2.0])

        result = rectified_flow_step(x_t, t=0.0, t_next=1.0, v_theta=v)
        # dt = 1.0, result = [2,2]
        assert np.allclose(result, np.array([2.0, 2.0]), atol=1e-10)

    def test_dtype_preservation(self):
        """输出 dtype 应与输入 x_t 一致。"""
        for dt in [np.float32, np.float64]:
            x_t = np.array([1.0, 2.0], dtype=dt)
            v = np.array([0.5, -0.5], dtype=dt)
            result = rectified_flow_step(x_t, t=1.0, t_next=0.0, v_theta=v)
            assert result.dtype == dt

    def test_multidimensional(self):
        """多维输入应正确工作。"""
        x_t = np.random.randn(4, 8, 8).astype(np.float64)
        v = np.random.randn(4, 8, 8).astype(np.float64)

        result = rectified_flow_step(x_t, t=1.0, t_next=0.5, v_theta=v)
        expected = x_t + (0.5 - 1.0) * v
        assert np.allclose(result, expected, atol=1e-10)
        assert result.shape == (4, 8, 8)


# ═══════════════════════════════════════════════════════════════════════════
# rectified_flow_sample 测试
# ═══════════════════════════════════════════════════════════════════════════

class TestRectifiedFlowSample:

    def test_constant_v_convergence_simple(self):
        """v = x 时，所有点应流向原点（dx/dt=x, t递减→x缩小）。"""
        np.random.seed(42)
        x_T = np.random.randn(50, 2).astype(np.float64)
        timesteps = np.linspace(1.0, 0.0, 29)  # 28 步

        def v_fn(x, t):
            return x  # dx/dt=x → x(t)=c·e^t，t 自 1→0 时 x 缩小

        x_0 = rectified_flow_sample(v_fn, x_T, timesteps)

        final_norms = np.linalg.norm(x_0, axis=1)
        initial_norms = np.linalg.norm(x_T, axis=1)
        assert np.all(final_norms < initial_norms * 0.5), \
            f"Some points did not converge. Max final norm: {final_norms.max()}"

    def test_known_flow_convergence(self):
        """
        v = x_1_known - x_0_known 常数向量场：
        从 x_T = x_1_known 出发，应精确到达 x_0_known。
        """
        np.random.seed(123)
        x_0_known = np.array([3.0, -2.0, 1.0], dtype=np.float64)
        x_1_known = np.array([-1.0, 4.0, 0.0], dtype=np.float64)
        v_known = x_1_known - x_0_known

        def v_fn(x, t):
            return v_known  # 常数向量场

        timesteps = np.linspace(1.0, 0.0, 101)  # 100 步
        x_0_result = rectified_flow_sample(v_fn, x_1_known, timesteps)

        assert np.allclose(x_0_result, x_0_known, atol=1e-10), \
            f"Expected {x_0_known}, got {x_0_result}"

    def test_zero_flow_no_change(self):
        """v=0 时，输出应等于输入。"""
        np.random.seed(42)
        x_T = np.random.randn(10, 3).astype(np.float64)
        timesteps = np.linspace(1.0, 0.0, 5)

        def v_fn(x, t):
            return np.zeros_like(x)

        x_0 = rectified_flow_sample(v_fn, x_T, timesteps)
        assert np.allclose(x_0, x_T, atol=1e-10)

    def test_reproducibility_same_seed(self):
        """相同 seed 和输入应产生相同输出。"""
        np.random.seed(42)
        x_T = np.random.randn(100, 4).astype(np.float64)
        timesteps = np.linspace(1.0, 0.0, 11)

        def v_fn(x, t):
            # 确定性向量场（无随机性）
            return -0.5 * x

        result1 = rectified_flow_sample(v_fn, x_T.copy(), timesteps, seed=0)
        result2 = rectified_flow_sample(v_fn, x_T.copy(), timesteps, seed=0)

        assert np.allclose(result1, result2, atol=1e-10)

    def test_callback_invoked(self):
        """callback 应在每步被调用。"""
        x_T = np.array([1.0, 0.0])
        timesteps = np.linspace(1.0, 0.0, 5)  # 4 步

        records = []

        def v_fn(x, t):
            return np.array([-0.5, -0.5])

        def cb(i, t, t_next, x_t):
            records.append((i, t, t_next))

        rectified_flow_sample(v_fn, x_T, timesteps, callback=cb)
        assert len(records) == 4, f"Expected 4 callback invocations, got {len(records)}"
        # 验证 i 从 0 到 3
        assert [r[0] for r in records] == [0, 1, 2, 3]

    def test_timesteps_length_error(self):
        """timesteps 只有 1 个点时抛异常。"""
        x_T = np.array([1.0, 0.0])
        timesteps = np.array([1.0])  # only 1 point

        def v_fn(x, t):
            return np.zeros_like(x)

        with pytest.raises(ValueError, match="timesteps"):
            rectified_flow_sample(v_fn, x_T, timesteps)

    def test_timesteps_non_monotonic_error(self):
        """非单调 timesteps 应抛异常。"""
        x_T = np.array([1.0, 0.0])
        timesteps = np.array([1.0, 0.5, 0.8, 0.0])  # non-monotonic

        def v_fn(x, t):
            return np.zeros_like(x)

        with pytest.raises(ValueError, match="monotonic|单调"):
            rectified_flow_sample(v_fn, x_T, timesteps)

    def test_edge_case_single_step(self):
        """timesteps=[1.0, 0.0]（单步）应正确工作。"""
        x_T = np.array([2.0, -1.0], dtype=np.float64)
        timesteps = np.array([1.0, 0.0])

        def v_fn(x, t):
            return np.array([-1.0, 1.0])

        x_0 = rectified_flow_sample(v_fn, x_T, timesteps)
        expected = x_T + (0.0 - 1.0) * np.array([-1.0, 1.0])
        assert np.allclose(x_0, expected, atol=1e-10)

    def test_dtype_preservation(self):
        """输出 dtype 应与输入 x_T 一致。"""
        for dt in [np.float32, np.float64]:
            x_T = np.random.randn(5, 3).astype(dt)
            timesteps = np.linspace(1.0, 0.0, 5)

            def v_fn(x, t):
                return np.zeros_like(x)

            x_0 = rectified_flow_sample(v_fn, x_T, timesteps)
            assert x_0.dtype == dt
