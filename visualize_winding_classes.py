"""
Compare random walks on the punctured plane grouped by winding number.

Runs many walks of fixed length N, records their final winding number,
then bins walks by integer winding w = -4, -3, -2, -1, 0, +1, +2, +3, +4.
For each winding class we compare:
  - Example trajectories
  - Spatial spread (max Chebyshev radius reached)
  - Scale penetration (s_min, s_max)
  - Fraction of time spent on the ring
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from src.walker import Walker
from src.graph import RING_CCW, is_ring

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_WINDINGS = [-4, -3, -2, -1, 1, 2, 3, 4]
N_STEPS = 20_000


def run_walks_and_bin(n_walks=5000, n_steps=N_STEPS, option='A', seed=42):
    """Run many walks, record stats, bin by integer winding.

    Returns dict mapping integer winding -> list of stat dicts.
    Also returns a dict mapping integer winding -> one example (xs, ys, ss).
    """
    rng = np.random.default_rng(seed)
    bins = {}       # w_int -> [stat_dicts]
    examples = {}   # w_int -> {'xs': ..., 'ys': ..., 'ss': ...}

    for i in range(n_walks):
        w = Walker(1, 0, 0, option=option,
                   rng=np.random.default_rng(rng.integers(2**62)))

        # Only record full path if we might need it as an example
        need_path = True  # record first few, then stop
        if need_path:
            xs, ys, ss = [w.x], [w.y], [w.s]

        x_max, y_max = 1, 0
        ring_steps = 0

        for step in range(1, n_steps + 1):
            w.step()
            ax, ay = abs(w.x), abs(w.y)
            if ax > x_max:
                x_max = ax
            if ay > y_max:
                y_max = ay
            if is_ring(w.x, w.y):
                ring_steps += 1
            if need_path:
                xs.append(w.x)
                ys.append(w.y)
                ss.append(w.s)

        # Bin by rounded winding number
        w_int = int(round(w.winding))
        stat = {
            'winding_exact': w.winding,
            'length': n_steps,
            'spread': max(x_max, y_max),
            's_min': w.s_min,
            's_max': w.s_max,
            's_range': w.s_max - w.s_min,
            'ring_frac': ring_steps / n_steps,
        }

        if w_int not in bins:
            bins[w_int] = []
        bins[w_int].append(stat)

        # Keep one example per winding class
        if w_int not in examples and need_path:
            examples[w_int] = {
                'xs': np.array(xs), 'ys': np.array(ys), 'ss': np.array(ss),
                'winding_exact': w.winding, 'spread': stat['spread'],
                's_min': stat['s_min'], 's_max': stat['s_max'],
            }

        if i % 500 == 0 and i > 0:
            print(f'    {i}/{n_walks} walks done', flush=True)

    return bins, examples


def plot_walk_path(ax, ex, winding):
    """Plot a single walk trajectory colored by time."""
    xs, ys = ex['xs'], ex['ys']
    n = len(xs)
    points = np.column_stack([xs, ys])
    segments = np.array([points[i:i+2] for i in range(n - 1)])
    t = np.linspace(0, 1, n - 1)
    lc = LineCollection(segments, cmap='viridis', linewidths=0.4, alpha=0.7)
    lc.set_array(t)
    ax.add_collection(lc)

    ax.plot(0, 0, 'x', color='red', markersize=8, markeredgewidth=2, zorder=5)
    ax.plot(xs[0], ys[0], 'o', color='lime', markersize=4, zorder=6,
            markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(xs[-1], ys[-1], 'D', color='magenta', markersize=4, zorder=6,
            markeredgecolor='black', markeredgewidth=0.5)

    pad = 2
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect('equal')
    w_exact = ex['winding_exact']
    ax.set_title(f'w={winding:+d} (w_exact={w_exact:+.2f}, spread={ex["spread"]:.0f})',
                 fontsize=8)
    ax.grid(True, alpha=0.15)


def main():
    t0 = time.time()
    n_walks = 6000
    print(f'Running {n_walks} walks of {N_STEPS} steps each (Option A)...\n')

    bins, examples = run_walks_and_bin(n_walks=n_walks, n_steps=N_STEPS,
                                       option='A', seed=314159)

    # Print winding distribution
    print(f'\nWinding distribution (N={N_STEPS}, {n_walks} walks):')
    for w_int in sorted(bins.keys()):
        print(f'  w={w_int:+d}: {len(bins[w_int])} walks')

    # --- Figure 1: Example paths (2 rows x 4 cols) ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Walks by Winding Number (N={N_STEPS}, Option A)\n'
                 f'Top: clockwise (w<0)   Bottom: counterclockwise (w>0)',
                 fontsize=13, fontweight='bold', y=0.99)

    top_row = [-4, -3, -2, -1]
    bot_row = [1, 2, 3, 4]

    for col, w_target in enumerate(top_row):
        ax = axes[0, col]
        if w_target in examples:
            plot_walk_path(ax, examples[w_target], w_target)
            count = len(bins.get(w_target, []))
            ax.text(0.02, 0.98, f'n={count}', transform=ax.transAxes,
                    va='top', fontsize=7, color='gray')
        else:
            ax.text(0.5, 0.5, f'w={w_target:+d}\nNo walks', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(f'w = {w_target:+d}', fontsize=9)

    for col, w_target in enumerate(bot_row):
        ax = axes[1, col]
        if w_target in examples:
            plot_walk_path(ax, examples[w_target], w_target)
            count = len(bins.get(w_target, []))
            ax.text(0.02, 0.98, f'n={count}', transform=ax.transAxes,
                    va='top', fontsize=7, color='gray')
        else:
            ax.text(0.5, 0.5, f'w={w_target:+d}\nNo walks', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(f'w = {w_target:+d}', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = OUTPUT_DIR / 'winding_excursion_paths.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {p1}')

    # --- Figure 2: Statistical comparison (4 panels) ---
    # Only plot for winding targets that have enough data
    plot_targets = [w for w in TARGET_WINDINGS if len(bins.get(w, [])) >= 3]
    labels = [f'{w:+d}' for w in plot_targets]

    n_neg = sum(1 for w in plot_targets if w < 0)
    n_pos = sum(1 for w in plot_targets if w > 0)
    colors_neg = plt.cm.Blues(np.linspace(0.8, 0.35, max(n_neg, 1)))
    colors_pos = plt.cm.Reds(np.linspace(0.35, 0.8, max(n_pos, 1)))
    colors = list(colors_neg[:n_neg]) + list(colors_pos[:n_pos])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Walk Statistics by Winding Number (N={N_STEPS}, Option A)',
                 fontsize=14, fontweight='bold', y=0.98)

    # (a) Spatial spread
    ax = axes[0, 0]
    spread_data = [[e['spread'] for e in bins[w]] for w in plot_targets]
    bp = ax.boxplot(spread_data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, data in enumerate(spread_data):
        med = np.median(data)
        ax.annotate(f'{med:.0f}', xy=(i + 1, med),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7)
    ax.set_ylabel('Max Chebyshev radius')
    ax.set_xlabel('Winding number w')
    ax.set_title('(a) Spatial Spread')
    ax.grid(True, alpha=0.2, axis='y')

    # (b) Scale penetration: s_min and s_max as box plots side by side
    ax = axes[0, 1]
    smin_data = [[e['s_min'] for e in bins[w]] for w in plot_targets]
    smax_data = [[e['s_max'] for e in bins[w]] for w in plot_targets]
    x_pos = np.arange(len(plot_targets))

    smin_med = [np.median(d) for d in smin_data]
    smax_med = [np.median(d) for d in smax_data]
    smin_q25 = [np.percentile(d, 25) for d in smin_data]
    smin_q75 = [np.percentile(d, 75) for d in smin_data]
    smax_q25 = [np.percentile(d, 25) for d in smax_data]
    smax_q75 = [np.percentile(d, 75) for d in smax_data]

    smin_med = np.array(smin_med, dtype=float)
    smax_med = np.array(smax_med, dtype=float)

    ax.bar(x_pos - 0.15, smax_med, width=0.3, color='firebrick', alpha=0.7,
           label='s_max (median)')
    ax.bar(x_pos + 0.15, smin_med, width=0.3, color='steelblue', alpha=0.7,
           label='s_min (median)')
    ax.errorbar(x_pos - 0.15, smax_med,
                yerr=[smax_med - smax_q25, np.array(smax_q75) - smax_med],
                fmt='none', color='firebrick', capsize=3, alpha=0.5)
    ax.errorbar(x_pos + 0.15, smin_med,
                yerr=[smin_med - np.array(smin_q25), np.array(smin_q75) - smin_med],
                fmt='none', color='steelblue', capsize=3, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Scale s')
    ax.set_xlabel('Winding number w')
    ax.set_title('(b) Scale Penetration (median, IQR)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # (c) Ring fraction
    ax = axes[1, 0]
    ring_data = [[e['ring_frac'] for e in bins[w]] for w in plot_targets]
    bp = ax.boxplot(ring_data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, data in enumerate(ring_data):
        med = np.median(data)
        ax.annotate(f'{med:.3f}', xy=(i + 1, med),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7)
    ax.set_ylabel('Fraction of steps on ring')
    ax.set_xlabel('Winding number w')
    ax.set_title('(c) Ring Occupation Fraction')
    ax.grid(True, alpha=0.2, axis='y')

    # (d) Spread and scale range vs |winding|, CW vs CCW overlay
    ax = axes[1, 1]
    for sign, color, marker, label in [(-1, 'steelblue', 'o', 'CW (w<0)'),
                                        (1, 'firebrick', 's', 'CCW (w>0)')]:
        abs_ws = sorted(set(abs(w) for w in plot_targets if np.sign(w) == sign))
        med_spread = []
        med_srange = []
        for aw in abs_ws:
            w_target = sign * aw
            data = bins.get(w_target, [])
            if data:
                med_spread.append(np.median([e['spread'] for e in data]))
                med_srange.append(np.median([e['s_range'] for e in data]))
            else:
                med_spread.append(np.nan)
                med_srange.append(np.nan)
        ax.plot(abs_ws, med_spread, f'{marker}-', color=color, linewidth=2,
                markersize=8, label=f'{label}: spread', zorder=3)
        ax.plot(abs_ws, med_srange, f'{marker}--', color=color, linewidth=1.5,
                markersize=6, alpha=0.6, label=f'{label}: scale range', zorder=3)

    ax.set_xlabel('|Winding number|')
    ax.set_ylabel('Median value')
    ax.set_title('(d) Spread & Scale Range vs |Winding|')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([1, 2, 3, 4])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = OUTPUT_DIR / 'winding_excursion_stats.png'
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p2}')

    # --- Summary table ---
    print('\n' + '=' * 90)
    print(f'{"Wind":>5} {"Count":>6} {"Med Spread":>11} {"Med s_min":>10} '
          f'{"Med s_max":>10} {"Med s_range":>12} {"Med ring%":>10}')
    print('-' * 90)
    for w_int in sorted(bins.keys()):
        data = bins[w_int]
        n = len(data)
        if n >= 1:
            print(f'{w_int:>+5d} {n:>6d} '
                  f'{np.median([e["spread"] for e in data]):>11.1f} '
                  f'{np.median([e["s_min"] for e in data]):>10.1f} '
                  f'{np.median([e["s_max"] for e in data]):>10.1f} '
                  f'{np.median([e["s_range"] for e in data]):>12.1f} '
                  f'{np.median([e["ring_frac"] for e in data]):>10.4f}')
    print('=' * 90)
    print(f'\nTotal time: {time.time() - t0:.1f}s')
    print('Done!')


if __name__ == '__main__':
    main()
