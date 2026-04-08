"""
Visualize random walk paths on the discrete punctured plane.

Generates a multi-panel figure showing:
  (a) Single walk trajectory colored by time step
  (b) Single walk in (x, y, s) 3D space
  (c) Multiple independent walks overlaid
  (d) Ring detail: close-up of the origin region showing chimney transitions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Circle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.walker import Walker
from src.graph import RING_CCW

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def record_walk(n_steps, option='A', seed=None):
    """Run a walk and record the full trajectory."""
    rng = np.random.default_rng(seed)
    w = Walker(x=1, y=0, s=0, option=option, rng=rng)
    xs, ys, ss = [w.x], [w.y], [w.s]
    for _ in range(n_steps):
        w.step()
        xs.append(w.x)
        ys.append(w.y)
        ss.append(w.s)
    return np.array(xs), np.array(ys), np.array(ss)


def plot_single_walk_2d(ax, xs, ys, ss, title='Walk on the punctured plane'):
    """Plot a single walk colored by time step."""
    n = len(xs)
    points = np.column_stack([xs, ys])
    segments = np.array([points[i:i+2] for i in range(n - 1)])
    t = np.linspace(0, 1, n - 1)
    lc = LineCollection(segments, cmap='viridis', linewidths=0.6, alpha=0.8)
    lc.set_array(t)
    ax.add_collection(lc)

    # Mark origin as punctured
    ax.plot(0, 0, 'x', color='red', markersize=10, markeredgewidth=2.5, zorder=5)

    # Mark ring positions
    ring_x = [p[0] for p in RING_CCW]
    ring_y = [p[1] for p in RING_CCW]
    ax.plot(ring_x, ring_y, 's', color='red', markersize=4, alpha=0.4, zorder=4)

    # Start and end
    ax.plot(xs[0], ys[0], 'o', color='lime', markersize=7, zorder=6,
            markeredgecolor='black', markeredgewidth=0.8, label='start')
    ax.plot(xs[-1], ys[-1], 'D', color='magenta', markersize=6, zorder=6,
            markeredgecolor='black', markeredgewidth=0.8, label='end')

    pad = 2
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(xs)))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Step', fontsize=8)


def plot_walk_3d(ax, xs, ys, ss, title='Walk in (x, y, scale) space'):
    """3D plot showing scale transitions."""
    n = len(xs)
    points = np.column_stack([xs, ys, ss]).astype(float)
    segments = [points[i:i+2] for i in range(n - 1)]
    t = np.linspace(0, 1, n - 1)

    lc = Line3DCollection(segments, cmap='plasma', linewidths=0.6, alpha=0.8)
    lc.set_array(t)
    ax.add_collection3d(lc)

    ax.set_xlim(xs.min() - 1, xs.max() + 1)
    ax.set_ylim(ys.min() - 1, ys.max() + 1)
    ax.set_zlim(ss.min() - 1, ss.max() + 1)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('scale s', fontsize=8)
    ax.set_title(title, fontsize=10)

    # Mark start
    ax.scatter([xs[0]], [ys[0]], [ss[0]], c='lime', s=40, edgecolors='black',
               linewidths=0.8, zorder=5)


def plot_multiple_walks(ax, n_walks=8, n_steps=2000, option='A'):
    """Overlay several walks with different colors."""
    cmap = matplotlib.colormaps['tab10']
    for i in range(n_walks):
        xs, ys, ss = record_walk(n_steps, option=option, seed=42 + i)
        color = cmap(i / n_walks)
        ax.plot(xs, ys, '-', color=color, linewidth=0.4, alpha=0.7)
        ax.plot(xs[0], ys[0], 'o', color=color, markersize=4,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.plot(xs[-1], ys[-1], 'D', color=color, markersize=3,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    ax.plot(0, 0, 'x', color='red', markersize=12, markeredgewidth=2.5, zorder=6)
    ring_x = [p[0] for p in RING_CCW]
    ring_y = [p[1] for p in RING_CCW]
    ax.plot(ring_x, ring_y, 's', color='red', markersize=4, alpha=0.4, zorder=4)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{n_walks} walks, N={n_steps} (Option {option})')
    ax.grid(True, alpha=0.15)


def plot_ring_detail(ax, xs, ys, ss, title='Ring region & scale transitions'):
    """Close-up of the ring region, color-coded by scale."""
    # Only plot points near the origin
    mask = (np.abs(xs) <= 4) & (np.abs(ys) <= 4)
    # Find contiguous segments within the mask
    n = len(xs)
    points = np.column_stack([xs, ys])

    # Color segments by scale
    s_min, s_max = ss.min(), ss.max()
    if s_min == s_max:
        s_max = s_min + 1
    norm = plt.Normalize(s_min, s_max)
    cmap = plt.cm.coolwarm

    for i in range(n - 1):
        if mask[i] and mask[i + 1]:
            color = cmap(norm(ss[i]))
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], '-',
                    color=color, linewidth=0.8, alpha=0.8)

    # Draw the ring
    ring_x = [p[0] for p in RING_CCW] + [RING_CCW[0][0]]
    ring_y = [p[1] for p in RING_CCW] + [RING_CCW[0][1]]
    ax.plot(ring_x, ring_y, '-', color='gray', linewidth=1, alpha=0.4, zorder=3)

    # Mark ring vertices with labels
    for idx, (rx, ry) in enumerate(RING_CCW):
        ax.plot(rx, ry, 'o', color='red', markersize=6, zorder=5)
        ax.annotate(str(idx), (rx, ry), textcoords="offset points",
                    xytext=(6, 6), fontsize=6, color='red')

    ax.plot(0, 0, 'x', color='red', markersize=14, markeredgewidth=3, zorder=6)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Scale s', fontsize=8)


def plot_scale_colored_walk(ax, xs, ys, ss, title='Walk colored by scale'):
    """Walk trajectory colored by the current scale."""
    n = len(xs)
    points = np.column_stack([xs, ys])
    segments = np.array([points[i:i+2] for i in range(n - 1)])
    s_colors = ss[:-1].astype(float)

    lc = LineCollection(segments, cmap='coolwarm', linewidths=0.6, alpha=0.8)
    lc.set_array(s_colors)
    ax.add_collection(lc)

    ax.plot(0, 0, 'x', color='red', markersize=10, markeredgewidth=2.5, zorder=5)
    ax.plot(xs[0], ys[0], 'o', color='lime', markersize=7, zorder=6,
            markeredgecolor='black', markeredgewidth=0.8)
    ax.plot(xs[-1], ys[-1], 'D', color='magenta', markersize=6, zorder=6,
            markeredgecolor='black', markeredgewidth=0.8)

    pad = 2
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(ss.min(), ss.max()))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Scale s', fontsize=8)


def main():
    print('Generating walk path visualizations...')
    N = 5000  # steps per walk

    # Record main walk (Option A)
    xs_a, ys_a, ss_a = record_walk(N, option='A', seed=12345)
    # Record Option B walk
    xs_b, ys_b, ss_b = record_walk(N, option='B', seed=12345)

    # --- Figure 1: 4-panel overview (Option A) ---
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('Random Walks on the Discrete Punctured Plane (Option A)',
                 fontsize=14, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(2, 2, 1)
    plot_single_walk_2d(ax1, xs_a, ys_a, ss_a,
                        title=f'(a) Single walk, N={N} (colored by time)')

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_walk_3d(ax2, xs_a, ys_a, ss_a,
                 title='(b) Walk in (x, y, scale) space')

    ax3 = fig.add_subplot(2, 2, 3)
    plot_multiple_walks(ax3, n_walks=10, n_steps=N, option='A')
    ax3.set_title(f'(c) 10 independent walks, N={N}')

    ax4 = fig.add_subplot(2, 2, 4)
    plot_ring_detail(ax4, xs_a, ys_a, ss_a,
                     title='(d) Ring region detail (colored by scale)')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / 'walk_paths_optA.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 2: Option A vs Option B comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Option A vs Option B: Walk Paths Comparison',
                 fontsize=14, fontweight='bold', y=0.98)

    plot_single_walk_2d(axes[0, 0], xs_a, ys_a, ss_a,
                        title=f'Option A: walk path (N={N})')
    plot_single_walk_2d(axes[0, 1], xs_b, ys_b, ss_b,
                        title=f'Option B: walk path (N={N})')
    plot_scale_colored_walk(axes[1, 0], xs_a, ys_a, ss_a,
                            title='Option A: colored by scale')
    plot_scale_colored_walk(axes[1, 1], xs_b, ys_b, ss_b,
                            title='Option B: colored by scale')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / 'walk_paths_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 3: Longer walk to show large-scale structure ---
    N_long = 20000
    xs_l, ys_l, ss_l = record_walk(N_long, option='A', seed=99)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Long Walk (N={N_long}, Option A)',
                 fontsize=14, fontweight='bold')

    plot_single_walk_2d(axes[0], xs_l, ys_l, ss_l,
                        title='Path colored by time')
    plot_scale_colored_walk(axes[1], xs_l, ys_l, ss_l,
                            title='Path colored by scale')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / 'walk_paths_long.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # Print summary stats
    print(f'\n--- Walk statistics (Option A, N={N}) ---')
    print(f'  x range: [{xs_a.min()}, {xs_a.max()}]')
    print(f'  y range: [{ys_a.min()}, {ys_a.max()}]')
    print(f'  scale range: [{ss_a.min()}, {ss_a.max()}]')
    print(f'  unique scales visited: {len(set(ss_a))}')

    print(f'\n--- Walk statistics (Option B, N={N}) ---')
    print(f'  x range: [{xs_b.min()}, {xs_b.max()}]')
    print(f'  y range: [{ys_b.min()}, {ys_b.max()}]')
    print(f'  scale range: [{ss_b.min()}, {ss_b.max()}]')
    print(f'  unique scales visited: {len(set(ss_b))}')

    print(f'\n--- Long walk (Option A, N={N_long}) ---')
    print(f'  x range: [{xs_l.min()}, {xs_l.max()}]')
    print(f'  y range: [{ys_l.min()}, {ys_l.max()}]')
    print(f'  scale range: [{ss_l.min()}, {ss_l.max()}]')

    print('\nDone! All visualizations saved to output/')


if __name__ == '__main__':
    main()
