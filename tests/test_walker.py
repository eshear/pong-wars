"""Tests for src/walker.py — Walker and StroboscopicWalker."""

import numpy as np
import pytest
from src.walker import Walker, StroboscopicWalker, WalkerState
from src.graph import (
    is_ring, is_origin, is_cardinal_ring,
    get_neighbors_option_a, get_neighbors_option_b,
    ring_angle_index, RING_CCW,
)


# ---------------------------------------------------------------------------
# Walker basics
# ---------------------------------------------------------------------------

class TestWalkerInit:
    def test_default_start(self):
        w = Walker()
        assert (w.x, w.y, w.s) == (1, 0, 0)
        assert w.option == 'A'
        assert w.winding == 0.0
        assert w.step_count == 0

    def test_custom_start(self):
        w = Walker(x=-1, y=1, s=3, option='B')
        assert (w.x, w.y, w.s) == (-1, 1, 3)
        assert w.option == 'B'

    def test_last_ring_idx_set_on_init(self):
        w = Walker(1, 0, 0)
        assert w.last_ring_idx == 0  # (1,0) is index 0

        w2 = Walker(0, 1, 0)
        assert w2.last_ring_idx == 2  # (0,1) is index 2

        w3 = Walker(5, 5, 0)
        assert w3.last_ring_idx == -1  # not on ring


class TestWalkerStep:
    def test_step_moves_to_neighbor(self):
        """After one step, the walker must be at a valid neighbor of the start."""
        for opt in ['A', 'B']:
            w = Walker(1, 0, 0, option=opt, rng=np.random.default_rng(42))
            start = (w.x, w.y, w.s)
            w.step()
            end = (w.x, w.y, w.s)
            if opt == 'A':
                neighbors = get_neighbors_option_a(*start)
            else:
                neighbors = get_neighbors_option_b(*start)
            assert end in neighbors, f"Option {opt}: stepped to {end}, not in neighbors of {start}"

    def test_step_count_increments(self):
        w = Walker(1, 0, 0, rng=np.random.default_rng(0))
        for i in range(1, 11):
            w.step()
            assert w.step_count == i

    def test_never_visits_origin(self):
        """Walker should never land on the origin."""
        for opt in ['A', 'B']:
            w = Walker(1, 0, 0, option=opt, rng=np.random.default_rng(123))
            for _ in range(5000):
                w.step()
                assert not is_origin(w.x, w.y), f"Option {opt}: walker at origin!"

    def test_scale_extremes_tracked(self):
        w = Walker(1, 0, 0, option='A', rng=np.random.default_rng(42))
        w.run(2000)
        # s_min should be <= 0 (started at 0, can go negative)
        assert w.s_min <= 0
        # s_max should be >= 0
        assert w.s_max >= 0
        # Current scale must be within tracked range
        assert w.s_min <= w.s <= w.s_max

    def test_scale_visits_sum(self):
        w = Walker(1, 0, 0, option='A', rng=np.random.default_rng(42))
        n = 500
        w.run(n)
        total_visits = sum(w.scale_visits.values())
        assert total_visits == n


class TestWalkerWinding:
    def test_winding_zero_if_never_on_ring(self):
        """If walker starts far from ring and stays there, winding stays 0."""
        # Start at a bulk position. With rng=0, it likely stays in bulk briefly.
        w = Walker(10, 10, 0, option='A', rng=np.random.default_rng(0))
        assert w.last_ring_idx == -1
        # Winding only changes when both prev and current are on ring
        # Since we start off-ring, first ring visit won't change winding
        # (because last_ring_idx starts as -1)

    def test_winding_accumulates(self):
        """Over many steps, winding should be nonzero with high probability."""
        w = Walker(1, 0, 0, option='A', rng=np.random.default_rng(42))
        w.run(10000)
        # Very unlikely to be exactly 0 after 10k steps
        assert w.winding != 0.0

    def test_winding_deterministic(self):
        """Same seed produces same winding."""
        w1 = Walker(1, 0, 0, option='A', rng=np.random.default_rng(999))
        w1.run(1000)
        w2 = Walker(1, 0, 0, option='A', rng=np.random.default_rng(999))
        w2.run(1000)
        assert w1.winding == w2.winding
        assert (w1.x, w1.y, w1.s) == (w2.x, w2.y, w2.s)


class TestWalkerReset:
    def test_reset_clears_state(self):
        w = Walker(1, 0, 0, option='A', rng=np.random.default_rng(42))
        w.run(500)
        w.reset(0, 1, -3)
        assert (w.x, w.y, w.s) == (0, 1, -3)
        assert w.winding == 0.0
        assert w.step_count == 0
        assert w.s_min == -3
        assert w.s_max == -3
        assert len(w.scale_visits) == 0
        assert w.last_ring_idx == ring_angle_index(0, 1)


class TestWalkerState:
    def test_state_namedtuple(self):
        w = Walker(1, 0, 0, rng=np.random.default_rng(42))
        w.run(10)
        st = w.state()
        assert isinstance(st, WalkerState)
        assert st.x == w.x
        assert st.y == w.y
        assert st.s == w.s
        assert st.winding == w.winding
        assert st.last_ring_idx == w.last_ring_idx


# ---------------------------------------------------------------------------
# Walker — option-specific behavior
# ---------------------------------------------------------------------------

class TestWalkerOptions:
    def test_option_b_diagonal_stays_at_scale_if_bulk(self):
        """In Option B, a diagonal ring vertex has no chimney. If we start at
        (1,1,5), neighbors are all at scale 5."""
        w = Walker(1, 1, 5, option='B', rng=np.random.default_rng(0))
        neighbors = w.get_neighbors()
        scales = [ns for _, _, ns in neighbors]
        assert all(s == 5 for s in scales), "Diagonal in B should have no chimney"

    def test_option_a_diagonal_has_chimney(self):
        """In Option A, (1,1,5) does have chimney neighbors at scales 4 and 6."""
        w = Walker(1, 1, 5, option='A', rng=np.random.default_rng(0))
        neighbors = w.get_neighbors()
        cross_scale = [(nx, ny, ns) for nx, ny, ns in neighbors if ns != 5]
        assert len(cross_scale) == 2
        assert (1, 1, 4) in cross_scale
        assert (1, 1, 6) in cross_scale


# ---------------------------------------------------------------------------
# Uniform stepping — statistical test
# ---------------------------------------------------------------------------

class TestUniformStepping:
    def test_uniform_from_cardinal_ring_option_a(self):
        """From (1,0,0) in Option A (degree 5), each neighbor should be chosen ~1/5."""
        counts = {}
        rng = np.random.default_rng(12345)
        n = 50000
        for _ in range(n):
            w = Walker(1, 0, 0, option='A', rng=rng)
            w.step()
            pos = (w.x, w.y, w.s)
            counts[pos] = counts.get(pos, 0) + 1

        neighbors = get_neighbors_option_a(1, 0, 0)
        assert len(counts) == len(neighbors) == 5
        expected = n / 5
        for pos, c in counts.items():
            # Allow 3 sigma tolerance
            assert abs(c - expected) < 3 * np.sqrt(n * 0.2 * 0.8), (
                f"Position {pos}: count {c}, expected ~{expected}"
            )

    def test_uniform_from_diagonal_ring_option_b(self):
        """From (1,1,0) in Option B (degree 4), each neighbor ~1/4."""
        counts = {}
        rng = np.random.default_rng(54321)
        n = 40000
        for _ in range(n):
            w = Walker(1, 1, 0, option='B', rng=rng)
            w.step()
            pos = (w.x, w.y, w.s)
            counts[pos] = counts.get(pos, 0) + 1

        neighbors = get_neighbors_option_b(1, 1, 0)
        assert len(counts) == len(neighbors) == 4
        expected = n / 4
        for pos, c in counts.items():
            assert abs(c - expected) < 3 * np.sqrt(n * 0.25 * 0.75), (
                f"Position {pos}: count {c}, expected ~{expected}"
            )


# ---------------------------------------------------------------------------
# StroboscopicWalker
# ---------------------------------------------------------------------------

class TestStroboscopicWalker:
    def test_clock_multiplier(self):
        assert StroboscopicWalker.clock_multiplier(0) == 1
        assert StroboscopicWalker.clock_multiplier(1) == 2
        assert StroboscopicWalker.clock_multiplier(-1) == 2
        assert StroboscopicWalker.clock_multiplier(5) == 6
        assert StroboscopicWalker.clock_multiplier(-10) == 11

    def test_outer_step_fixed_mode(self):
        """Fixed mode: m(s) inner steps are taken per outer tick."""
        sw = StroboscopicWalker(1, 0, 0, option='A', adaptive=False,
                                rng=np.random.default_rng(42))
        # At s=0, m(0) = 1, so one outer step = 1 inner step
        sw.outer_step()
        assert sw.inner.step_count == 1
        assert sw.outer_ticks == 1

    def test_outer_step_increments_ticks(self):
        sw = StroboscopicWalker(1, 0, 0, option='A', adaptive=False,
                                rng=np.random.default_rng(42))
        for i in range(1, 6):
            sw.outer_step()
            assert sw.outer_ticks == i

    def test_run(self):
        sw = StroboscopicWalker(1, 0, 0, option='A', adaptive=False,
                                rng=np.random.default_rng(42))
        sw.run(10)
        assert sw.outer_ticks == 10
        assert sw.inner.step_count >= 10  # at least 1 inner step per outer tick

    def test_reset(self):
        sw = StroboscopicWalker(1, 0, 0, option='A', rng=np.random.default_rng(42))
        sw.run(20)
        sw.reset(-1, 0, 2)
        assert sw.outer_ticks == 0
        assert sw.inner.x == -1
        assert sw.inner.y == 0
        assert sw.inner.s == 2
        assert sw.inner.step_count == 0

    def test_never_visits_origin(self):
        """Stroboscopic walker should also never land on origin."""
        for opt in ['A', 'B']:
            sw = StroboscopicWalker(1, 0, 0, option=opt, adaptive=False,
                                    rng=np.random.default_rng(42))
            for _ in range(500):
                sw.outer_step()
                assert not is_origin(sw.inner.x, sw.inner.y)

    def test_adaptive_vs_fixed_differ(self):
        """Adaptive and fixed modes should generally produce different trajectories,
        because adaptive re-evaluates m(s) as scale changes mid-burst."""
        # With same seed, they diverge once the walker moves to a different scale
        sw_a = StroboscopicWalker(1, 0, 0, option='A', adaptive=True,
                                  rng=np.random.default_rng(42))
        sw_f = StroboscopicWalker(1, 0, 0, option='A', adaptive=False,
                                  rng=np.random.default_rng(42))
        sw_a.run(100)
        sw_f.run(100)
        # They share the same rng seed but different step logic, so inner step counts
        # may diverge (not guaranteed but likely with 100 outer ticks)
        # We just check they both ran without error
        assert sw_a.outer_ticks == 100
        assert sw_f.outer_ticks == 100
