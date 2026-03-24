"""Tests for src/graph.py — graph structure and adjacency."""

import pytest
from src.graph import (
    RING_CCW, RING_INDEX, DIRECTIONS,
    is_ring, is_origin, is_cardinal_ring,
    get_neighbors_option_a, get_neighbors_option_b,
    ring_angle_index, angular_displacement,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_ring_ccw_length(self):
        assert len(RING_CCW) == 8

    def test_ring_ccw_all_chebyshev_1(self):
        for x, y in RING_CCW:
            assert max(abs(x), abs(y)) == 1

    def test_ring_ccw_no_origin(self):
        assert (0, 0) not in RING_CCW

    def test_ring_index_bijection(self):
        """RING_INDEX maps each of the 8 ring positions to a unique index 0–7."""
        assert len(RING_INDEX) == 8
        assert set(RING_INDEX.values()) == set(range(8))
        for pos, idx in RING_INDEX.items():
            assert RING_CCW[idx] == pos

    def test_directions_are_four_cardinal(self):
        assert set(DIRECTIONS) == {(1, 0), (-1, 0), (0, 1), (0, -1)}


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

class TestPredicates:
    @pytest.mark.parametrize("x,y", RING_CCW)
    def test_is_ring_true(self, x, y):
        assert is_ring(x, y)

    @pytest.mark.parametrize("x,y", [(0, 0), (2, 0), (0, 2), (3, 3), (-2, 1)])
    def test_is_ring_false(self, x, y):
        assert not is_ring(x, y)

    def test_is_origin(self):
        assert is_origin(0, 0)
        assert not is_origin(1, 0)
        assert not is_origin(0, 1)

    @pytest.mark.parametrize("x,y", [(1, 0), (-1, 0), (0, 1), (0, -1)])
    def test_is_cardinal_ring_true(self, x, y):
        assert is_cardinal_ring(x, y)

    @pytest.mark.parametrize("x,y", [(1, 1), (-1, 1), (-1, -1), (1, -1)])
    def test_is_cardinal_ring_false_for_diagonals(self, x, y):
        assert not is_cardinal_ring(x, y)

    @pytest.mark.parametrize("x,y", [(0, 0), (2, 0), (3, 3)])
    def test_is_cardinal_ring_false_for_non_ring(self, x, y):
        assert not is_cardinal_ring(x, y)


# ---------------------------------------------------------------------------
# Neighbor lists — structural invariants
# ---------------------------------------------------------------------------

def _no_origin_in_neighbors(neighbors):
    """Assert the origin never appears in a neighbor list."""
    for nx, ny, ns in neighbors:
        assert (nx, ny) != (0, 0), f"Origin (0,0,{ns}) found in neighbors"


def _all_neighbors_are_lattice_adjacent_or_chimney(x, y, s, neighbors):
    """Each neighbor is either a 4-adjacent lattice vertex (same scale) or chimney (same x,y, adjacent scale)."""
    for nx, ny, ns in neighbors:
        if ns == s:
            # In-scale: must be 4-adjacent
            assert abs(nx - x) + abs(ny - y) == 1, f"({nx},{ny},{ns}) not 4-adjacent to ({x},{y},{s})"
        else:
            # Cross-scale: chimney — same (x,y), scale differs by 1
            assert (nx, ny) == (x, y), f"Chimney neighbor ({nx},{ny},{ns}) has different (x,y)"
            assert abs(ns - s) == 1, f"Chimney neighbor ({nx},{ny},{ns}) scale differs by {abs(ns-s)}"


class TestOptionA:
    """Option A: chimney chains for all 8 ring positions (additional edges)."""

    @pytest.mark.parametrize("x,y", [(1, 0), (-1, 0), (0, 1), (0, -1)])
    def test_cardinal_ring_degree_5(self, x, y):
        neighbors = get_neighbors_option_a(x, y, 0)
        assert len(neighbors) == 5

    @pytest.mark.parametrize("x,y", [(1, 1), (-1, 1), (-1, -1), (1, -1)])
    def test_diagonal_ring_degree_6(self, x, y):
        neighbors = get_neighbors_option_a(x, y, 0)
        assert len(neighbors) == 6

    def test_bulk_degree_4(self):
        """A vertex far from origin has 4 neighbors."""
        assert len(get_neighbors_option_a(5, 5, 0)) == 4
        assert len(get_neighbors_option_a(2, 3, -7)) == 4

    def test_bulk_adjacent_to_origin_degree_3(self):
        """(2,0) has a 4-neighbor at (1,0) but also would go to origin via (-1,0) direction.
        Wait — (2,0) neighbors are (3,0),(1,0),(2,1),(2,-1). None are origin. Degree 4."""
        # Bulk vertex (2,0): its lattice neighbors are (3,0),(1,0),(2,1),(2,-1) — none are origin
        assert len(get_neighbors_option_a(2, 0, 0)) == 4

    @pytest.mark.parametrize("x,y", RING_CCW)
    def test_no_origin_in_ring_neighbors(self, x, y):
        _no_origin_in_neighbors(get_neighbors_option_a(x, y, 0))

    @pytest.mark.parametrize("x,y", RING_CCW)
    def test_ring_has_chimney_up_and_down(self, x, y):
        neighbors = get_neighbors_option_a(x, y, 5)
        chimney = [(nx, ny, ns) for nx, ny, ns in neighbors if ns != 5]
        assert sorted(chimney) == sorted([(x, y, 4), (x, y, 6)])

    def test_adjacency_structure(self):
        """All neighbors are lattice-adjacent or chimney-adjacent."""
        for x, y in RING_CCW:
            for s in [-3, 0, 7]:
                neighbors = get_neighbors_option_a(x, y, s)
                _all_neighbors_are_lattice_adjacent_or_chimney(x, y, s, neighbors)

    def test_no_self_loops(self):
        for x, y in RING_CCW:
            for s in [-2, 0, 3]:
                neighbors = get_neighbors_option_a(x, y, s)
                assert (x, y, s) not in neighbors

    def test_scale_independence(self):
        """Degree doesn't depend on which scale we're at."""
        for x, y in RING_CCW:
            d0 = len(get_neighbors_option_a(x, y, 0))
            for s in [-100, -1, 1, 100]:
                assert len(get_neighbors_option_a(x, y, s)) == d0


class TestOptionB:
    """Option B: chimney chains for 4 cardinal directions only."""

    @pytest.mark.parametrize("x,y", [(1, 0), (-1, 0), (0, 1), (0, -1)])
    def test_cardinal_ring_degree_5(self, x, y):
        neighbors = get_neighbors_option_b(x, y, 0)
        assert len(neighbors) == 5

    @pytest.mark.parametrize("x,y", [(1, 1), (-1, 1), (-1, -1), (1, -1)])
    def test_diagonal_ring_degree_4(self, x, y):
        neighbors = get_neighbors_option_b(x, y, 0)
        assert len(neighbors) == 4

    def test_bulk_degree_4(self):
        assert len(get_neighbors_option_b(5, 5, 0)) == 4
        assert len(get_neighbors_option_b(2, 3, -7)) == 4

    @pytest.mark.parametrize("x,y", [(1, 0), (-1, 0), (0, 1), (0, -1)])
    def test_cardinal_has_chimney_up_and_down(self, x, y):
        neighbors = get_neighbors_option_b(x, y, 5)
        chimney = [(nx, ny, ns) for nx, ny, ns in neighbors if ns != 5]
        assert sorted(chimney) == sorted([(x, y, 4), (x, y, 6)])

    @pytest.mark.parametrize("x,y", [(1, 1), (-1, 1), (-1, -1), (1, -1)])
    def test_diagonal_has_no_chimney(self, x, y):
        neighbors = get_neighbors_option_b(x, y, 5)
        chimney = [(nx, ny, ns) for nx, ny, ns in neighbors if ns != 5]
        assert chimney == []

    @pytest.mark.parametrize("x,y", RING_CCW)
    def test_no_origin_in_ring_neighbors(self, x, y):
        _no_origin_in_neighbors(get_neighbors_option_b(x, y, 0))

    def test_adjacency_structure(self):
        for x, y in RING_CCW:
            for s in [-3, 0, 7]:
                neighbors = get_neighbors_option_b(x, y, s)
                _all_neighbors_are_lattice_adjacent_or_chimney(x, y, s, neighbors)

    def test_no_self_loops(self):
        for x, y in RING_CCW:
            for s in [-2, 0, 3]:
                neighbors = get_neighbors_option_b(x, y, s)
                assert (x, y, s) not in neighbors

    def test_scale_independence(self):
        for x, y in RING_CCW:
            d0 = len(get_neighbors_option_b(x, y, 0))
            for s in [-100, -1, 1, 100]:
                assert len(get_neighbors_option_b(x, y, s)) == d0


class TestUndirectedness:
    """If v is in neighbors(u), then u must be in neighbors(v)."""

    @pytest.mark.parametrize("get_neighbors", [get_neighbors_option_a, get_neighbors_option_b])
    def test_undirected_ring_vertices(self, get_neighbors):
        """Check symmetry for ring vertices at several scales."""
        for x, y in RING_CCW:
            for s in [-2, 0, 3]:
                neighbors = get_neighbors(x, y, s)
                for nx, ny, ns in neighbors:
                    reverse = get_neighbors(nx, ny, ns)
                    assert (x, y, s) in reverse, (
                        f"({x},{y},{s}) -> ({nx},{ny},{ns}) but not reverse "
                        f"in {get_neighbors.__name__}"
                    )

    @pytest.mark.parametrize("get_neighbors", [get_neighbors_option_a, get_neighbors_option_b])
    def test_undirected_bulk_vertices(self, get_neighbors):
        for x, y, s in [(3, 4, 0), (10, -7, 2), (-5, 3, -1)]:
            neighbors = get_neighbors(x, y, s)
            for nx, ny, ns in neighbors:
                reverse = get_neighbors(nx, ny, ns)
                assert (x, y, s) in reverse


# ---------------------------------------------------------------------------
# Angular displacement
# ---------------------------------------------------------------------------

class TestAngularDisplacement:
    def test_identity(self):
        for i in range(8):
            assert angular_displacement(i, i) == 0

    def test_one_step_ccw(self):
        for i in range(8):
            assert angular_displacement(i, (i + 1) % 8) == 1

    def test_one_step_cw(self):
        for i in range(8):
            assert angular_displacement(i, (i - 1) % 8) == -1

    def test_half_turn(self):
        # Half turn (displacement 4) is ambiguous in direction; function returns 4
        assert abs(angular_displacement(0, 4)) == 4
        assert abs(angular_displacement(4, 0)) == 4

    def test_range(self):
        for i in range(8):
            for j in range(8):
                d = angular_displacement(i, j)
                assert -4 <= d <= 4

    def test_antisymmetry(self):
        """displacement(a,b) = -displacement(b,a) except at ±4."""
        for i in range(8):
            for j in range(8):
                d_fwd = angular_displacement(i, j)
                d_rev = angular_displacement(j, i)
                if abs(d_fwd) < 4:
                    assert d_fwd == -d_rev


class TestRingAngleIndex:
    @pytest.mark.parametrize("pos,idx", list(RING_INDEX.items()))
    def test_known_positions(self, pos, idx):
        assert ring_angle_index(*pos) == idx

    @pytest.mark.parametrize("x,y", [(0, 0), (2, 0), (3, 3)])
    def test_non_ring_returns_neg1(self, x, y):
        assert ring_angle_index(x, y) == -1
