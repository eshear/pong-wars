# Random Walks on the Discrete Punctured Plane

## March 2026

### 1. The Graph

**1.1. The Bulk Lattice**
At each integer scale s in Z, there is a copy of the punctured square lattice Z^2 \ {(0,0)}.
Vertex set at scale s: Vs = { (x, y, s) : x, y in Z, (x, y) != (0, 0) }

Within each scale s, edges connect the usual 4-neighbors: (x, y, s) is adjacent to (x+/-1, y, s)
and (x, y+/-1, s), provided both endpoints are in Vs.

**1.2. The Ring**
The 8 lattice points adjacent to the origin at scale s form the ring Rs:
Rs = { (x, y, s) : max(|x|, |y|) = 1 }

These are the 8 points: (+/-1, 0, s), (0, +/-1, s), (+/-1, +/-1, s).
The ring is a cycle C8 in the natural adjacency order: label them 0 through 7 going
counterclockwise from (1, 0).

**1.3. The Chimney (Inter-Scale Edges)**
When a walker at scale s attempts to step from a ring vertex into the deleted origin, it
instead transitions to scale s-1. Concretely, for each ring vertex r in Rs, there is an edge
connecting r to the corresponding ring vertex in Rs-1. The correspondence is the identity
map on (x, y) coordinates.

Conversely, when a walker at scale s-1 steps outward from a ring vertex, it transitions
to scale s.

**1.4. Stepping Rule**

Option A (uniform on graph neighbors): Ring vertex has 4 in-scale neighbors + 1 chimney
neighbor = degree 5. Choose uniformly among 5.

Option B (origin-replacement): Treat chimney edge as replacing the deleted edge to origin.
Ring vertex has 4 neighbors: 2 ring, 1 bulk outward, 1 chimney downward. Choose uniformly
among 4.

### 2. The Stroboscopic Clock

Clock multiplier at scale s: m(s) = max(1, |s| + 1)

One outer tick = m(s) inner steps. Critical depth at s = -8 where 9 hidden steps alias a
full C8 revolution to a null step.

**2.1. Helix Interpretation**: n -> (n mod 8, floor(n/8)) gives angular position and scale.
Chimney is topologically Z, not C8 x Z.

**2.2. Thermal Regime**: At |s| ~ 8, walker has enough hidden steps to leave ring and
explore bulk. At |s| >> 8, walker spends most hidden steps lost in bulk. Chimney is
expected to be self-truncating.

### 3. Experiments

See implementation for full experiment list:
- 3.1: Basic walk statistics (no stroboscope)
- 3.2: Stroboscopic walk statistics
- 3.3: The aliasing transition (critical depth s = -8)
- 3.4: Comparison with hyperbolic random walk
- 4: Connection to Propp pi-estimation (excursion ratios)

### 4. Key Questions

- Is the walk recurrent or transient?
- Does E[8/T] for winding excursions equal pi/4?
- Does winding +2 vs +1 produce pi vs ln(2) dichotomy?
- What is the spectral dimension of G?
- Does the aliasing transition have period exactly 8?
