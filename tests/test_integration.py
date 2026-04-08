"""Integration tests — end-to-end walks and multi-step invariants."""

import numpy as np
import pytest
from src.walker import Walker, StroboscopicWalker
from src.graph import (
    is_origin, is_ring, is_cardinal_ring,
    get_neighbors_option_a, get_neighbors_option_b,
)


class TestLongWalkInvariants:
    """Run longer walks and verify structural invariants hold throughout."""

    @pytest.mark.parametrize("option", ['A', 'B'])
    def test_every_step_is_valid_neighbor(self, option):
        """Trace a 2000-step walk and verify every step is to a valid neighbor."""
        get_neighbors = get_neighbors_option_a if option == 'A' else get_neighbors_option_b
        w = Walker(1, 0, 0, option=option, rng=np.random.default_rng(77))
        for _ in range(2000):
            px, py, ps = w.x, w.y, w.s
            w.step()
            neighbors = get_neighbors(px, py, ps)
            assert (w.x, w.y, w.s) in neighbors

    @pytest.mark.parametrize("option", ['A', 'B'])
    def test_origin_never_visited(self, option):
        w = Walker(1, 0, 0, option=option, rng=np.random.default_rng(88))
        for _ in range(5000):
            w.step()
            assert not is_origin(w.x, w.y)

    @pytest.mark.parametrize("option", ['A', 'B'])
    def test_scale_extremes_monotonic(self, option):
        """s_min is non-increasing and s_max is non-decreasing over time."""
        w = Walker(1, 0, 0, option=option, rng=np.random.default_rng(99))
        prev_smin = 0
        prev_smax = 0
        for _ in range(1000):
            w.step()
            assert w.s_min <= prev_smin
            assert w.s_max >= prev_smax
            prev_smin = w.s_min
            prev_smax = w.s_max


class TestOptionBScaleTrapping:
    """In Option B, diagonal ring vertices have no chimney.
    A walker that enters a diagonal ring position can only leave via in-scale moves."""

    def test_diagonal_never_changes_scale_directly(self):
        """If walker is at diagonal ring vertex in Option B, stepping cannot change scale."""
        rng = np.random.default_rng(42)
        # Manually check: from (1,1,s), all neighbors in Option B are at same scale
        for s in [-5, 0, 3]:
            neighbors = get_neighbors_option_b(1, 1, s)
            for nx, ny, ns in neighbors:
                assert ns == s, f"Diagonal (1,1,{s}) has neighbor at scale {ns}"


class TestChimneyTraversal:
    """Verify walkers actually use chimney edges."""

    def test_option_a_reaches_multiple_scales(self):
        """Option A walker should reach multiple scales in 5000 steps."""
        w = Walker(1, 0, 0, option='A', rng=np.random.default_rng(42))
        w.run(5000)
        scale_range = w.s_max - w.s_min
        assert scale_range >= 2, f"Scale range {scale_range} too small for Option A"

    def test_option_b_reaches_multiple_scales(self):
        """Option B walker should also reach multiple scales (via cardinal chimneys)."""
        w = Walker(1, 0, 0, option='B', rng=np.random.default_rng(42))
        w.run(5000)
        scale_range = w.s_max - w.s_min
        assert scale_range >= 2, f"Scale range {scale_range} too small for Option B"

    def test_option_b_scale_changes_only_at_cardinal(self):
        """In Option B, scale changes can only happen at cardinal ring positions."""
        w = Walker(1, 0, 0, option='B', rng=np.random.default_rng(42))
        for _ in range(5000):
            prev_x, prev_y, prev_s = w.x, w.y, w.s
            w.step()
            if w.s != prev_s:
                # Scale changed — the previous position must have been a cardinal ring vertex
                assert is_cardinal_ring(prev_x, prev_y), (
                    f"Scale changed from ({prev_x},{prev_y},{prev_s}) to s={w.s}, "
                    f"but ({prev_x},{prev_y}) is not cardinal ring"
                )


class TestStroboscopicIntegration:
    @pytest.mark.parametrize("option", ['A', 'B'])
    def test_stroboscopic_never_visits_origin(self, option):
        sw = StroboscopicWalker(1, 0, 0, option=option, adaptive=False,
                                rng=np.random.default_rng(42))
        for _ in range(200):
            sw.outer_step()
            assert not is_origin(sw.inner.x, sw.inner.y)

    def test_inner_steps_ge_outer_ticks(self):
        """Inner step count >= outer tick count (m(s) >= 1 always)."""
        sw = StroboscopicWalker(1, 0, 0, option='A', adaptive=False,
                                rng=np.random.default_rng(42))
        sw.run(50)
        assert sw.inner.step_count >= 50


class TestWindingConsistency:
    """Verify winding tracking is consistent with angular positions."""

    def test_full_turn_winding(self):
        """If we manually walk around the ring CCW, winding should be +1."""
        # This is tricky because we can't control the random walk.
        # Instead, verify the math: 8 steps of +1/8 each = 1.0 full turn.
        from src.graph import angular_displacement
        total = 0
        for i in range(8):
            total += angular_displacement(i, (i + 1) % 8) / 8.0
        assert abs(total - 1.0) < 1e-12

    def test_winding_unchanged_by_scale_change(self):
        """Chimney moves (same x,y, different s) don't change ring angle index,
        so they shouldn't produce spurious winding."""
        from src.graph import ring_angle_index, angular_displacement
        # (1,0) at any scale has the same ring_angle_index = 0
        assert ring_angle_index(1, 0) == 0
        # A chimney step from (1,0,0) to (1,0,1) moves to same ring position
        # angular_displacement(0, 0) = 0, so winding doesn't change
        assert angular_displacement(0, 0) == 0
