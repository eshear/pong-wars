"""
The discrete punctured plane graph G.

Vertex: (x, y, s) where (x,y) in Z^2 \ {(0,0)}, s in Z.
Ring vertices: max(|x|, |y|) == 1 (the 8 neighbors of origin).
Chimney edges connect ring vertices across scales.
"""

import numpy as np
from typing import List, Tuple

# The 8 ring positions in counterclockwise order starting from (1,0)
RING_CCW = [
    (1, 0),   # 0
    (1, 1),   # 1
    (0, 1),   # 2
    (-1, 1),  # 3
    (-1, 0),  # 4
    (-1, -1), # 5
    (0, -1),  # 6
    (1, -1),  # 7
]

# Map (x,y) -> ring index
RING_INDEX = {pos: i for i, pos in enumerate(RING_CCW)}

# 4-connected neighbors (lattice moves)
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def is_ring(x: int, y: int) -> bool:
    """Check if (x,y) is a ring vertex (adjacent to origin in Chebyshev sense)."""
    return max(abs(x), abs(y)) == 1


def is_origin(x: int, y: int) -> bool:
    return x == 0 and y == 0


def is_cardinal_ring(x: int, y: int) -> bool:
    """Check if (x,y) is a cardinal ring vertex (4-neighbor of origin)."""
    return (x, y) in {(1, 0), (-1, 0), (0, 1), (0, -1)}


def get_neighbors_option_a(x: int, y: int, s: int) -> List[Tuple[int, int, int]]:
    """
    Option A: chimney chains for all 8 ring positions (additional edges).

    On 4-connected Z² with the origin deleted at every scale:
      Cardinal ring vertices lose 1 neighbor (the origin) -> 3 in-scale
      Diagonal ring vertices lose 0 neighbors -> 4 in-scale

    Chimney edges are ADDITIONAL for all 8 ring positions, connecting
    (x,y,s) to (x,y,s-1) and (x,y,s+1) via an undirected chain.

    Resulting degrees:
      Cardinal ring: 3 in-scale + 2 chimney = 5
      Diagonal ring: 4 in-scale + 2 chimney = 6
      Bulk: 4 (or 3 if adjacent to deleted origin)
    """
    neighbors = []

    # In-scale lattice neighbors (origin always deleted)
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if not is_origin(nx, ny):
            neighbors.append((nx, ny, s))

    # Chimney edges for ALL 8 ring positions — both up and down
    if is_ring(x, y):
        neighbors.append((x, y, s - 1))
        neighbors.append((x, y, s + 1))

    return neighbors


def get_neighbors_option_b(x: int, y: int, s: int) -> List[Tuple[int, int, int]]:
    """
    Option B: chimney chains for 4 cardinal directions only.

    On 4-connected Z² with the origin deleted at every scale:
      Cardinal ring vertices lose 1 neighbor (the origin) -> 3 in-scale
      Diagonal ring vertices lose 0 neighbors -> 4 in-scale

    For each cardinal direction, a single undirected infinite chain connects
    ...(x,y,s+1) <-> (x,y,s) <-> (x,y,s-1)...
    This chain replaces the deleted origin edge at each scale.
    Diagonal ring vertices have NO chimney edges.

    Resulting degrees:
      Cardinal ring: 3 in-scale + 2 chimney = 5
      Diagonal ring: 4 in-scale + 0 chimney = 4
      Bulk: 4 (or 3 if adjacent to deleted origin)
    """
    neighbors = []

    # In-scale lattice neighbors (origin always deleted)
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if not is_origin(nx, ny):
            neighbors.append((nx, ny, s))

    # Chimney edges for CARDINAL ring positions only — both up and down
    if is_cardinal_ring(x, y):
        neighbors.append((x, y, s - 1))
        neighbors.append((x, y, s + 1))

    return neighbors


def ring_angle_index(x: int, y: int) -> int:
    """Return the ring index (0-7) for a ring vertex, or -1 if not on ring."""
    if not is_ring(x, y):
        return -1
    return RING_INDEX.get((x, y), -1)


def angular_displacement(idx_from: int, idx_to: int) -> int:
    """
    Compute signed angular displacement on the C8 ring.
    Returns value in {-4, -3, -2, -1, 0, 1, 2, 3, 4}.
    Positive = counterclockwise.
    """
    diff = (idx_to - idx_from) % 8
    if diff > 4:
        diff -= 8
    return diff
