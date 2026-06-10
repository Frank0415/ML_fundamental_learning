"""
test_scheduler.py — EulerScheduler 与 RectifiedFlowScheduler 的 pytest 测试

覆盖：
- 常数向量场下 Euler 与 RectifiedFlow 的一致性
- num_steps=1 时单步更新
- num_steps=0 抛 ValueError
- sigma / timestep 边界情况：单调性
- deterministic seed：相同 seed 产生相同结果
"""

import numpy as np
import pytest
from diffusion_engine.core.scheduler import EulerScheduler, RectifiedFlowScheduler


# ── 辅助函数 ────────────────────────────────────────────────────────────

def constant_vector_field(x):
    """一个简单的常数向量场：v = [1.0, 1.0]（恒定方向）。"""
    v = np.ones_like(x, dtype=x.dtype)
    return v


def zero_vector_field(x):
    """v = 0，不应改变位置。"""
    return np.zeros_like(x, dtype=x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# EulerScheduler 测试
# ═══════════════════════════════════════════════════════════════════════════

class TestEulerScheduler:

    def test_constant_v_linear_motion(self):
        """常数向量场 v=[1,1] 时，x_t 应沿直线匀速运动。"""
        sched = EulerScheduler(num_steps=10, sigma_min=0.01, sigma_max=1.0)
        x0 = np.array([0.0, 0.0], dtype=np.float64)

        x_t = x0.copy()
        v = constant_vector_field(x_t)
        for i in range(len(sched.sigmas) - 1):
            x_t = sched.step(v, x_t, i)

        # 总位移应等于 sigma 总变化 × v = (sigma_min - sigma_max) × [1,1]
        expected = x0 + (sched.sigmas[-1] - sched.sigmas[0]) * np.ones(2)
        assert np.allclose(x_t, expected, atol=1e-10), \
            f"Expected {expected}, got {x_t}"

    def test_num_steps_one(self):
        """num_steps=1 时，单步更新应正确。"""
        sched = EulerScheduler(num_steps=1, sigma_min=0.01, sigma_max=1.0)
        x0 = np.array([2.0, 3.0], dtype=np.float64)
        v = np.array([0.5, -0.5], dtype=np.float64)

        x_t = sched.step(v, x0, 0)  # i=0, 唯一一步
        # dt = sigma[1] - sigma[0] = 0.01 - 1.0 = -0.99
        expected = x0 + (0.01 - 1.0) * v
        assert np.allclose(x_t, expected, atol=1e-10)

    def test_num_steps_zero_raises(self):
        """num_steps=0 应抛 ValueError。"""
        with pytest.raises(ValueError, match="num_steps"):
            EulerScheduler(num_steps=0)

    def test_num_steps_negative_raises(self):
        """num_steps=-1 应抛 ValueError。"""
        with pytest.raises(ValueError, match="num_steps"):
            EulerScheduler(num_steps=-1)

    def test_sigmas_monotonic_descending(self):
        """get_sigmas 返回的 sigma 序列应是单调递减的。"""
        sigmas = EulerScheduler.get_sigmas(num_steps=50)
        diffs = np.diff(sigmas)
        assert np.all(diffs < 0), f"sigmas must be strictly decreasing, diffs={diffs}"

    def test_sigmas_monotonic_small(self):
        """仅 2 步的 sigma 也应单调递减。"""
        sigmas = EulerScheduler.get_sigmas(num_steps=2)
        assert sigmas[0] > sigmas[1]

    def test_deterministic_seed_same_result(self):
        """相同 seed 应产生相同结果（对 stochastic 模式）。"""
        sched1 = EulerScheduler(num_steps=5, seed=42, stochastic=True)
        sched2 = EulerScheduler(num_steps=5, seed=42, stochastic=True)

        x0 = np.array([1.0, 0.0])
        v = np.array([0.1, 0.1])

        x1 = x0.copy()
        for i in range(len(sched1.sigmas) - 1):
            x1 = sched1.step(v, x1, i)

        x2 = x0.copy()
        for i in range(len(sched2.sigmas) - 1):
            x2 = sched2.step(v, x2, i)

        assert np.allclose(x1, x2, atol=1e-10), \
            f"Same seed should give identical results: {x1} vs {x2}"

    def test_custom_sigmas(self):
        """自定义 sigma 序列应正确工作。"""
        custom = np.array([10.0, 5.0, 1.0, 0.0])
        sched = EulerScheduler(sigmas=custom)
        assert len(sched.sigmas) == 4
        x0 = np.array([0.0, 0.0])
        v = np.array([1.0, 2.0])

        x_t = x0.copy()
        for i in range(3):
            x_t = sched.step(v, x_t, i)

        # total dt = (0 - 10) = -10
        expected = x0 + (-10.0) * v
        assert np.allclose(x_t, expected, atol=1e-10)

    def test_step_index_out_of_range(self):
        """步索引越界应抛异常。"""
        sched = EulerScheduler(num_steps=5)
        x0 = np.array([0.0])
        v = np.array([1.0])
        with pytest.raises(IndexError):
            sched.step(v, x0, 100)

    def test_dtype_preservation(self):
        """输出 dtype 应与输入 x_t 一致。"""
        sched = EulerScheduler(num_steps=5)
        for dt in [np.float32, np.float64]:
            x0 = np.array([1.0, 2.0], dtype=dt)
            v = np.array([0.1, 0.2], dtype=dt)
            result = sched.step(v, x0, 0)
            assert result.dtype == dt

    def test_set_timesteps(self):
        """set_timesteps 应正确更新步数（num_steps=ODE步数, sigma点=num_steps+1）。"""
        sched = EulerScheduler(num_steps=10)
        assert len(sched.sigmas) == 11  # 10 步 = 11 个 sigma 点
        sched.set_timesteps(5)
        assert len(sched.sigmas) == 6


# ═══════════════════════════════════════════════════════════════════════════
# RectifiedFlowScheduler 测试
# ═══════════════════════════════════════════════════════════════════════════

class TestRectifiedFlowScheduler:

    def test_constant_v_pushes_toward_origin(self):
        """v = x 时（rectified flow 中推至原点），所有点应变小。"""
        sched = RectifiedFlowScheduler(num_steps=20, t_start=1.0, t_end=0.0)
        x0 = np.array([2.0, -3.0], dtype=np.float64)

        x_t = x0.copy()
        for i in range(sched.num_steps):
            v = x_t  # rectified flow 中 v=x 时 dx/dt=x → x(t)=c·e^t，t 减小则 x 缩小
            x_t = sched.step(v, x_t, i)

        assert np.linalg.norm(x_t) < np.linalg.norm(x0) * 0.5, \
            f"After integration, x_t should be closer to origin. norm={np.linalg.norm(x_t)}"

    def test_num_steps_one(self):
        """num_steps=1 时单步更新。"""
        sched = RectifiedFlowScheduler(num_steps=1, t_start=1.0, t_end=0.0)
        x0 = np.array([2.0, 3.0])
        v = np.array([1.0, -1.0])

        x_t = sched.step(v, x0, 0)
        # dt = 0.0 - 1.0 = -1.0
        expected = x0 + (-1.0) * v
        assert np.allclose(x_t, expected, atol=1e-10)

    def test_num_steps_zero_raises(self):
        """num_steps=0 应抛 ValueError。"""
        with pytest.raises(ValueError, match="num_steps"):
            RectifiedFlowScheduler(num_steps=0)

    def test_timesteps_monotonic(self):
        """默认 timesteps 应从 1.0 递减到 0.0。"""
        sched = RectifiedFlowScheduler(num_steps=28)
        assert sched.timesteps[0] == 1.0
        assert sched.timesteps[-1] == 0.0
        diffs = np.diff(sched.timesteps)
        assert np.all(diffs < 0), "timesteps must be decreasing"

    def test_timesteps_length(self):
        """timesteps 应有 num_steps + 1 个点。"""
        for n in [1, 5, 28, 100]:
            sched = RectifiedFlowScheduler(num_steps=n)
            assert len(sched.timesteps) == n + 1

    def test_set_timesteps(self):
        """set_timesteps 应正确更新步数和序列。"""
        sched = RectifiedFlowScheduler(num_steps=28)
        assert sched.num_steps == 28
        sched.set_timesteps(10)
        assert sched.num_steps == 10
        assert len(sched.timesteps) == 11

    def test_custom_t_range(self):
        """自定义 t_start/t_end 应正常工作。"""
        sched = RectifiedFlowScheduler(num_steps=5, t_start=2.0, t_end=-1.0)
        assert sched.timesteps[0] == 2.0
        assert sched.timesteps[-1] == -1.0

    def test_step_index_out_of_range(self):
        """步索引越界应抛异常。"""
        sched = RectifiedFlowScheduler(num_steps=5)
        x0 = np.array([0.0])
        v = np.array([1.0])
        with pytest.raises(IndexError):
            sched.step(v, x0, 100)

    def test_dtype_preservation(self):
        """输出 dtype 应与输入 x_t 一致。"""
        sched = RectifiedFlowScheduler(num_steps=5)
        for dt in [np.float32, np.float64]:
            x0 = np.array([1.0, 2.0], dtype=dt)
            v = np.array([0.1, 0.2], dtype=dt)
            result = sched.step(v, x0, 0)
            assert result.dtype == dt


# ═══════════════════════════════════════════════════════════════════════════
# 一致性测试
# ═══════════════════════════════════════════════════════════════════════════

class TestSchedulerConsistency:

    def test_euler_rf_equivalent_with_linear_schedule(self):
        """
        若 EulerScheduler 用线性 sigma（同 RectifiedFlow 的 timesteps），
        两者在 v=常数 时应产生完全相同的结果。
        """
        # 用相同的线性间隔点
        steps = np.linspace(1.0, 0.0, 21)

        euler = EulerScheduler(sigmas=steps)
        rf = RectifiedFlowScheduler(num_steps=20, t_start=1.0, t_end=0.0)

        x0 = np.array([2.0, -1.0], dtype=np.float64)
        v_constant = np.array([0.5, 0.5], dtype=np.float64)

        x_euler = x0.copy()
        x_rf = x0.copy()

        # Euler 用 sigma 列表（与 timesteps 相同值）
        for i in range(20):
            x_euler = euler.step(v_constant, x_euler, i)

        # RectifiedFlow 用 timesteps
        for i in range(20):
            x_rf = rf.step(v_constant, x_rf, i)

        # 总 dt 相同（1.0 → 0.0），v 相同 → 结果应一致
        assert np.allclose(x_euler, x_rf, atol=1e-10), \
            f"Euler={x_euler} vs RF={x_rf}"
