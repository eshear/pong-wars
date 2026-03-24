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


def get_neighbors_option_a(x: int, y: int, s: int) -> List[Tuple[int, int, int]]:
    """
    Option A: uniform on all graph neighbors.

    The chimney edge is an ADDITIONAL edge beyond the in-scale lattice edges.
    Each ring vertex (x,y,s) has one undirected chimney edge connecting it to (x,y,s-1).
    Since the graph is undirected, from (x,y,s) you can reach both (x,y,s-1) and (x,y,s+1)
    via chimney edges (your own edge to s-1, and s+1's edge down to you).

    Degrees:
      Cardinal ring vertex (e.g. (1,0)): 3 in-scale (origin deleted) + 2 chimney = 5
      Diagonal ring vertex (e.g. (1,1)): 4 in-scale + 2 chimney = 6
      Bulk vertex: 4 (standard lattice, or 3 if adjacent to origin)
    """
    neighbors = []

    # In-scale lattice neighbors (origin always deleted)
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if not is_origin(nx, ny):
            neighbors.append((nx, ny, s))

    # Chimney edges (for ring vertices only) — both up and down
    if is_ring(x, y):
        neighbors.append((x, y, s - 1))
        neighbors.append((x, y, s + 1))

    return neighbors


def get_neighbors_option_b(x: int, y: int, s: int) -> List[Tuple[int, int, int]]:
    """
    Option B: origin-replacement.

    The chimney replaces the deleted origin edge rather than being additional.
    For cardinal ring vertices that had an edge to origin: the chimney down to s-1
    takes that slot. The undirected chimney up to s+1 also exists (from s+1's
    perspective, their chimney down reaches us).

    For cardinal ring vertices: instead of losing the origin edge and gaining 2 chimney
    edges (as in Option A), the origin edge is replaced by chimney-down, plus chimney-up
    from undirectedness. Net: 3 in-scale + 1 chimney_down (replacing origin) + 1 chimney_up = 5.

    Hmm, that's still 5. The key difference from the spec is that Option B should give
    degree 4 for cardinal ring vertices. This means in Option B, only the DOWN chimney
    is available as a choice (the up direction is not an explicit neighbor — you can only
    arrive from above, not choose to go up). This creates asymmetric transition probs.

    Interpretation that matches spec's "degree 4":
      Cardinal: 3 in-scale + 1 chimney (down only, replacing origin) = 4
      Diagonal: 4 in-scale + 1 chimney (down only) = 5

    The walker can still go UP because from scale s+1, the ring vertex there has a
    chimney down to scale s. By undirectedness, the walker at s can traverse that edge
    upward. BUT in Option B, the walker at s does NOT see s+1 as a neighbor to choose.
    Instead, only from s+1 can you choose to go down to s.

    This is a DIRECTED interpretation. Let's implement it: Option B ring vertices only
    have chimney-down as a chooseable neighbor. Upward movement happens only when you're
    at s-1 and you chimney-down... wait, that's still downward.

    OK, simplest resolution: the graph has one chimney edge per ring vertex to s-1.
    In Option A, the graph is undirected so both endpoints can traverse it (degree +2).
    In Option B, the edge replaces origin and is also undirected, but only counts as +1
    because it's a replacement, not an addition. The degree difference:
      Option A cardinal: 3 + 2 = 5
      Option B cardinal: 3 + 1 = 4  (chimney occupies the slot of the deleted origin)

    We achieve degree-4 by only listing chimney-down (not chimney-up) for Option B.
    The walker can still go up because at the scale above, Option B also has chimney-down
    which is the same edge — so upward movement IS possible, just with different probability
    weighting since it's chosen from the upper scale's neighbor list, not the lower's.
    """
    neighbors = []

    # In-scale lattice neighbors (origin always deleted)
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if not is_origin(nx, ny):
            neighbors.append((nx, ny, s))

    # Chimney: DOWN only for Option B (replaces the origin edge)
    if is_ring(x, y):
        neighbors.append((x, y, s - 1))

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
