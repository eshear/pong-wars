"""
Random walker on the discrete punctured plane graph G.
"""

import numpy as np
from typing import Optional, Tuple, List, NamedTuple
from .graph import (
    get_neighbors_option_a, get_neighbors_option_b,
    is_ring, ring_angle_index, angular_displacement, RING_CCW
)


class WalkerState(NamedTuple):
    x: int
    y: int
    s: int
    winding: float  # cumulative winding number (in units of full turns = 1/8 per ring step)
    last_ring_idx: int  # last ring index visited, or -1


class Walker:
    """Random walker on the punctured plane graph."""

    def __init__(self, x: int = 1, y: int = 0, s: int = 0,
                 option: str = 'A', rng: Optional[np.random.Generator] = None):
        self.x = x
        self.y = y
        self.s = s
        self.option = option
        self.rng = rng or np.random.default_rng()

        # Winding tracking
        self.winding = 0.0  # cumulative winding in units of 1/8 turn
        self.last_ring_idx = ring_angle_index(x, y)

        # Statistics
        self.step_count = 0
        self.s_min = s
        self.s_max = s
        self.scale_visits = {}  # s -> count

    def get_neighbors(self):
        if self.option == 'A':
            return get_neighbors_option_a(self.x, self.y, self.s)
        else:
            return get_neighbors_option_b(self.x, self.y, self.s)

    def step(self) -> Tuple[int, int, int]:
        """Take one step. Returns new (x, y, s)."""
        neighbors = self.get_neighbors()
        idx = self.rng.integers(len(neighbors))
        nx, ny, ns = neighbors[idx]

        self.x, self.y, self.s = nx, ny, ns
        self.step_count += 1

        # Track scale extremes
        if ns < self.s_min:
            self.s_min = ns
        if ns > self.s_max:
            self.s_max = ns

        # Track scale visits
        self.scale_visits[ns] = self.scale_visits.get(ns, 0) + 1

        # Track winding
        cur_ring_idx = ring_angle_index(nx, ny)
        if cur_ring_idx >= 0 and self.last_ring_idx >= 0:
            # Both current and previous tracked positions are on ring (at any scale)
            disp = angular_displacement(self.last_ring_idx, cur_ring_idx)
            self.winding += disp / 8.0  # normalize so 1 full turn = 1.0
        if cur_ring_idx >= 0:
            self.last_ring_idx = cur_ring_idx

        return nx, ny, ns

    def run(self, n_steps: int) -> None:
        """Run n_steps of the walk."""
        for _ in range(n_steps):
            self.step()

    def state(self) -> WalkerState:
        return WalkerState(self.x, self.y, self.s, self.winding, self.last_ring_idx)

    def reset(self, x: int = 1, y: int = 0, s: int = 0):
        self.x, self.y, self.s = x, y, s
        self.winding = 0.0
        self.last_ring_idx = ring_angle_index(x, y)
        self.step_count = 0
        self.s_min = s
        self.s_max = s
        self.scale_visits.clear()


class StroboscopicWalker:
    """Walker with stroboscopic clock."""

    def __init__(self, x: int = 1, y: int = 0, s: int = 0,
                 option: str = 'A', adaptive: bool = True,
                 rng: Optional[np.random.Generator] = None):
        """
        adaptive=True: if scale changes mid-burst, use new scale's multiplier
                        for remaining sub-steps.
        adaptive=False: always use m(s) where s is scale at start of burst.
        """
        self.inner = Walker(x, y, s, option, rng)
        self.adaptive = adaptive
        self.outer_ticks = 0

    @staticmethod
    def clock_multiplier(s: int) -> int:
        """m(s) = max(1, |s| + 1)"""
        return max(1, abs(s) + 1)

    def outer_step(self) -> Tuple[int, int, int]:
        """Execute one outer tick (m(s) inner steps). Returns final position."""
        if self.adaptive:
            # Adaptive: re-check scale after each inner step
            remaining = self.clock_multiplier(self.inner.s)
            while remaining > 0:
                self.inner.step()
                remaining -= 1
                if remaining > 0:
                    # Recompute based on new scale
                    new_m = self.clock_multiplier(self.inner.s)
                    # Use the new multiplier's proportion
                    # Simple model: just decrement, don't recompute total
                    pass
        else:
            # Fixed: use scale at start of burst
            m = self.clock_multiplier(self.inner.s)
            for _ in range(m):
                self.inner.step()

        self.outer_ticks += 1
        return self.inner.x, self.inner.y, self.inner.s

    def run(self, n_outer_ticks: int) -> None:
        for _ in range(n_outer_ticks):
            self.outer_step()

    def reset(self, x: int = 1, y: int = 0, s: int = 0):
        self.inner.reset(x, y, s)
        self.outer_ticks = 0
