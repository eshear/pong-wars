#!/usr/bin/env python3
"""
Winding-count stopping criteria analysis using numba-JIT compiled walks.

Run random walks on the punctured plane that stop when the cumulative winding
number first reaches ±W for a target W.  Uses a single-walk approach:
run each walk until it either reaches |winding| >= max_target or times out,
and record the first-passage time for each target along the way.

For each target we collect:
  - Hitting time (number of steps to reach |winding| >= W)
  - Direction of winding at stopping (CW vs CCW)
  - Scale penetration (s_min, s_max)
  - Spatial spread (max Chebyshev radius)
  - Fraction of time on ring
"""

import numpy as np
import numba as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# Ring positions in CCW order: (1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)
RING_X = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
RING_Y = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

# Directions for 4-connected lattice
DIR_X = np.array([1, -1, 0, 0], dtype=np.int64)
DIR_Y = np.array([0, 0, 1, -1], dtype=np.int64)


@nb.njit(cache=True)
def _is_ring(x, y):
    return max(abs(x), abs(y)) == 1


@nb.njit(cache=True)
def _is_cardinal_ring(x, y):
    return (abs(x) + abs(y)) == 1 and _is_ring(x, y)


@nb.njit(cache=True)
def _ring_index(x, y):
    """Return ring index 0-7, or -1 if not on ring."""
    if not _is_ring(x, y):
        return -1
    for i in range(8):
        if RING_X[i] == x and RING_Y[i] == y:
            return i
    return -1


@nb.njit(cache=True)
def _angular_disp(idx_from, idx_to):
    """Signed angular displacement on C8 ring, in {-4,...,4}."""
    diff = (idx_to - idx_from) % 8
    if diff > 4:
        diff -= 8
    return diff


@nb.njit(cache=True)
def _walk_until_winding(max_target, max_steps, option_a, seed):
    """
    Run a single walk from (1,0,s=0) until |winding| >= max_target or timeout.

    Returns:
      winding_at_step: array of shape (max_target,) — step at which |winding| first >= t+1
                       (0 means not reached)
      winding_val:     array of shape (max_target,) — winding value at that step
      s_min, s_max:    scale extremes at time of reaching each target
      spread:          max Chebyshev radius at time of reaching each target
      ring_frac:       fraction of steps on ring at time of reaching each target
    """
    np.random.seed(seed)

    x, y, s = np.int64(1), np.int64(0), np.int64(0)
    winding = 0.0
    last_ring_idx = _ring_index(x, y)
    s_min_cur, s_max_cur = s, s
    x_max, y_max = abs(x), abs(y)
    ring_steps = 0
    next_target = 1  # next winding target to record

    # Output arrays
    hit_step = np.zeros(max_target, dtype=np.int64)
    hit_winding = np.zeros(max_target, dtype=np.float64)
    hit_s_min = np.zeros(max_target, dtype=np.int64)
    hit_s_max = np.zeros(max_target, dtype=np.int64)
    hit_spread = np.zeros(max_target, dtype=np.int64)
    hit_ring_frac = np.zeros(max_target, dtype=np.float64)

    # Pre-allocate neighbor buffer (max 6 neighbors for option A diagonal ring)
    nbr_x = np.empty(6, dtype=np.int64)
    nbr_y = np.empty(6, dtype=np.int64)
    nbr_s = np.empty(6, dtype=np.int64)

    for step in range(1, max_steps + 1):
        # Build neighbor list inline
        n_nbr = 0
        for d in range(4):
            nx = x + DIR_X[d]
            ny = y + DIR_Y[d]
            if nx == 0 and ny == 0:
                continue
            nbr_x[n_nbr] = nx
            nbr_y[n_nbr] = ny
            nbr_s[n_nbr] = s
            n_nbr += 1

        # Chimney edges
        if option_a:
            if _is_ring(x, y):
                nbr_x[n_nbr] = x; nbr_y[n_nbr] = y; nbr_s[n_nbr] = s - 1; n_nbr += 1
                nbr_x[n_nbr] = x; nbr_y[n_nbr] = y; nbr_s[n_nbr] = s + 1; n_nbr += 1
        else:
            if _is_cardinal_ring(x, y):
                nbr_x[n_nbr] = x; nbr_y[n_nbr] = y; nbr_s[n_nbr] = s - 1; n_nbr += 1
                nbr_x[n_nbr] = x; nbr_y[n_nbr] = y; nbr_s[n_nbr] = s + 1; n_nbr += 1

        # Random choice
        idx = np.random.randint(n_nbr)
        x, y, s = nbr_x[idx], nbr_y[idx], nbr_s[idx]

        # Track stats
        if s < s_min_cur:
            s_min_cur = s
        if s > s_max_cur:
            s_max_cur = s
        ax, ay = abs(x), abs(y)
        if ax > x_max:
            x_max = ax
        if ay > y_max:
            y_max = ay
        if _is_ring(x, y):
            ring_steps += 1

        # Track winding
        cur_ring_idx = _ring_index(x, y)
        if cur_ring_idx >= 0 and last_ring_idx >= 0:
            disp = _angular_disp(last_ring_idx, cur_ring_idx)
            winding += disp / 8.0
        if cur_ring_idx >= 0:
            last_ring_idx = cur_ring_idx

        # Check targets
        while next_target <= max_target and abs(winding) >= next_target - 1e-9:
            t_idx = next_target - 1
            hit_step[t_idx] = step
            hit_winding[t_idx] = winding
            hit_s_min[t_idx] = s_min_cur
            hit_s_max[t_idx] = s_max_cur
            hit_spread[t_idx] = max(x_max, y_max)
            hit_ring_frac[t_idx] = ring_steps / step
            next_target += 1

        if next_target > max_target:
            break

    return hit_step, hit_winding, hit_s_min, hit_s_max, hit_spread, hit_ring_frac


def run_batch(n_trials, max_target, max_steps, option_a, base_seed=42):
    """Run n_trials walks and collect results for all winding targets 1..max_target."""
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**62, size=n_trials)

    # Warm up JIT
    _walk_until_winding(1, 10, option_a, 0)

    # Results per target
    all_hit_steps = [[] for _ in range(max_target)]
    all_hit_winding = [[] for _ in range(max_target)]
    all_s_min = [[] for _ in range(max_target)]
    all_s_max = [[] for _ in range(max_target)]
    all_spread = [[] for _ in range(max_target)]
    all_ring_frac = [[] for _ in range(max_target)]

    t0 = time.time()
    for i in range(n_trials):
        hs, hw, hsn, hsx, hsp, hrf = _walk_until_winding(
            max_target, max_steps, option_a, int(seeds[i]))

        for t in range(max_target):
            if hs[t] > 0:
                all_hit_steps[t].append(hs[t])
                all_hit_winding[t].append(hw[t])
                all_s_min[t].append(hsn[t])
                all_s_max[t].append(hsx[t])
                all_spread[t].append(hsp[t])
                all_ring_frac[t].append(hrf[t])

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f'  {i+1}/{n_trials} trials ({rate:.1f}/s), '
                  f'hits: {[len(s) for s in all_hit_steps]}', flush=True)

    results = {}
    opt_label = 'A' if option_a else 'B'
    for t in range(max_target):
        wt = t + 1
        n_completed = len(all_hit_steps[t])
        ht = np.array(all_hit_steps[t]) if n_completed > 0 else np.array([], dtype=np.int64)
        fw = np.array(all_hit_winding[t]) if n_completed > 0 else np.array([])

        n_pos = int(np.sum(fw > 0)) if n_completed > 0 else 0
        n_neg = int(np.sum(fw < 0)) if n_completed > 0 else 0

        results[wt] = {
            'winding_target': wt,
            'option': opt_label,
            'n_trials': n_trials,
            'n_completed': n_completed,
            'n_timeout': n_trials - n_completed,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'hitting_times': ht,
            'final_windings': fw,
            's_mins': np.array(all_s_min[t]),
            's_maxs': np.array(all_s_max[t]),
            'spreads': np.array(all_spread[t]),
            'ring_fracs': np.array(all_ring_frac[t]),
        }
    return results


def print_summary(res):
    wt = res['winding_target']
    opt = res['option']
    nc = res['n_completed']
    nt = res['n_trials']
    print(f'\n{"="*70}')
    print(f'Winding stopping at |W| >= {wt}, Option {opt}')
    print(f'{"="*70}')
    print(f'  Completed: {nc}/{nt} ({100*nc/nt:.1f}%), Timeouts: {res["n_timeout"]}')
    print(f'  Direction: CCW(+) {res["n_positive"]}, CW(-) {res["n_negative"]}')

    if nc > 0:
        ht = res['hitting_times']
        print(f'\n  Hitting time (steps to reach |W|>={wt}):')
        print(f'    Mean:   {np.mean(ht):.1f}')
        print(f'    Median: {np.median(ht):.1f}')
        print(f'    Std:    {np.std(ht):.1f}')
        print(f'    Min:    {np.min(ht)}')
        print(f'    Max:    {np.max(ht)}')
        print(f'    Q25:    {np.percentile(ht, 25):.0f}')
        print(f'    Q75:    {np.percentile(ht, 75):.0f}')

        print(f'\n  Scale penetration:')
        print(f'    E[s_min]: {np.mean(res["s_mins"]):.2f}, '
              f'Med: {np.median(res["s_mins"]):.1f}')
        print(f'    E[s_max]: {np.mean(res["s_maxs"]):.2f}, '
              f'Med: {np.median(res["s_maxs"]):.1f}')
        print(f'    E[s_range]: {np.mean(res["s_maxs"] - res["s_mins"]):.2f}')

        print(f'\n  Spatial spread (max Chebyshev radius):')
        print(f'    Mean: {np.mean(res["spreads"]):.1f}, '
              f'Median: {np.median(res["spreads"]):.1f}')

        print(f'\n  Ring occupation:')
        print(f'    Mean ring frac: {np.mean(res["ring_fracs"]):.4f}')


def save_json_summary(res, label):
    summary = {
        'winding_target': res['winding_target'],
        'option': res['option'],
        'n_trials': res['n_trials'],
        'n_completed': res['n_completed'],
        'n_timeout': res['n_timeout'],
        'n_positive': res['n_positive'],
        'n_negative': res['n_negative'],
        'completion_rate': res['n_completed'] / res['n_trials'],
    }
    if res['n_completed'] > 0:
        ht = res['hitting_times']
        summary['hitting_time'] = {
            'mean': float(np.mean(ht)),
            'median': float(np.median(ht)),
            'std': float(np.std(ht)),
            'min': int(np.min(ht)),
            'max': int(np.max(ht)),
            'q25': float(np.percentile(ht, 25)),
            'q75': float(np.percentile(ht, 75)),
        }
        summary['scale'] = {
            's_min_mean': float(np.mean(res['s_mins'])),
            's_max_mean': float(np.mean(res['s_maxs'])),
            's_range_mean': float(np.mean(res['s_maxs'] - res['s_mins'])),
            's_min_median': float(np.median(res['s_mins'])),
            's_max_median': float(np.median(res['s_maxs'])),
        }
        summary['spatial'] = {
            'spread_mean': float(np.mean(res['spreads'])),
            'spread_median': float(np.median(res['spreads'])),
        }
        summary['ring_frac_mean'] = float(np.mean(res['ring_fracs']))

    path = OUTPUT_DIR / f'winding_stopping_{label}_summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {path}')


def plot_single_target(res, prefix='winding_stop'):
    wt = res['winding_target']
    opt = res['option']
    nc = res['n_completed']
    if nc == 0:
        print(f'  No completed walks for W={wt}, Option {opt}. Skipping plots.')
        return

    ht = res['hitting_times']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Winding Stopping Analysis: |W| >= {wt}, Option {opt}\n'
                 f'({nc}/{res["n_trials"]} walks completed)',
                 fontsize=13, fontweight='bold', y=0.99)

    # (a) Hitting time distribution
    ax = axes[0, 0]
    ax.hist(ht, bins=60, density=True, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(np.mean(ht), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(ht):.0f}')
    ax.axvline(np.median(ht), color='orange', linestyle='--', linewidth=1.5,
               label=f'Median = {np.median(ht):.0f}')
    ax.set_xlabel('Hitting time T (steps)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Hitting Time Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (b) Hitting time log-log survival
    ax = axes[0, 1]
    sorted_ht = np.sort(ht)
    survival = 1 - np.arange(1, len(sorted_ht) + 1) / len(sorted_ht)
    ax.loglog(sorted_ht, survival, '.', markersize=2, color='steelblue')
    if len(sorted_ht) > 10:
        log_ht = np.log(sorted_ht[10:])
        log_surv = np.log(survival[10:len(sorted_ht)])
        if len(log_ht) > 2 and len(log_surv) > 2:
            q1, q3 = len(log_ht)//4, 3*len(log_ht)//4
            if q3 > q1 + 2:
                slope, intercept = np.polyfit(log_ht[q1:q3], log_surv[q1:q3], 1)
                t_ref = np.logspace(np.log10(sorted_ht[10]),
                                    np.log10(sorted_ht[-2]), 50)
                ax.loglog(t_ref, np.exp(intercept) * t_ref**slope,
                          'r--', alpha=0.7, label=f'slope ~ {slope:.2f}')
                ax.legend(fontsize=8)
    ax.set_xlabel('Hitting time T')
    ax.set_ylabel('P(hitting time > T)')
    ax.set_title('(b) Hitting Time Survival (log-log)')
    ax.grid(True, alpha=0.2, which='both')

    # (c) Scale penetration
    ax = axes[1, 0]
    ax.hist(res['s_mins'], bins=40, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', label=f's_min (med={np.median(res["s_mins"]):.0f})')
    ax.hist(res['s_maxs'], bins=40, density=True, alpha=0.6, color='firebrick',
            edgecolor='black', label=f's_max (med={np.median(res["s_maxs"]):.0f})')
    ax.set_xlabel('Scale s')
    ax.set_ylabel('Density')
    ax.set_title('(c) Scale Penetration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (d) Ring occupation vs hitting time scatter
    ax = axes[1, 1]
    ax.scatter(ht, res['ring_fracs'], s=3, alpha=0.3, color='teal')
    ax.set_xlabel('Hitting time T')
    ax.set_ylabel('Ring fraction')
    ax.set_title('(d) Ring Occupation vs Hitting Time')
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'{prefix}_w{wt}_opt{opt}_overview'
    path = OUTPUT_DIR / f'{fname}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_comparison(results_by_target, opt_label, prefix='winding_stop'):
    """Compare multiple winding targets for one option."""
    targets = sorted(results_by_target.keys())
    results_list = [results_by_target[t] for t in targets
                    if results_by_target[t]['n_completed'] > 0]
    if len(results_list) < 2:
        print('  Need at least 2 result sets with completions for comparison.')
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Winding Stopping Comparison, Option {opt_label}',
                 fontsize=14, fontweight='bold', y=0.99)

    colors = ['steelblue', 'firebrick', 'seagreen', 'darkorange', 'purple']

    # (a) Hitting time distributions
    ax = axes[0, 0]
    for i, res in enumerate(results_list):
        wt = res['winding_target']
        ht = res['hitting_times']
        ax.hist(ht, bins=60, density=True, alpha=0.4, color=colors[i % len(colors)],
                edgecolor='black', linewidth=0.3,
                label=f'|W|>={wt} (n={res["n_completed"]}, med={np.median(ht):.0f})')
    ax.set_xlabel('Hitting time T (steps)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Hitting Time Distributions')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (b) Survival curves
    ax = axes[0, 1]
    for i, res in enumerate(results_list):
        wt = res['winding_target']
        ht = np.sort(res['hitting_times'])
        surv = 1 - np.arange(1, len(ht) + 1) / len(ht)
        ax.loglog(ht, surv, '.', markersize=2, color=colors[i % len(colors)],
                  label=f'|W|>={wt}')
    ax.set_xlabel('Hitting time T')
    ax.set_ylabel('P(T > t)')
    ax.set_title('(b) Survival Curves (log-log)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, which='both')

    # (c) Scale penetration box plots
    ax = axes[0, 2]
    positions = []
    box_data = []
    box_labels = []
    box_colors_list = []
    for i, res in enumerate(results_list):
        wt = res['winding_target']
        positions.extend([3*i + 0.8, 3*i + 1.6])
        box_data.extend([res['s_mins'], res['s_maxs']])
        box_labels.extend([f'W>={wt}\ns_min', f'W>={wt}\ns_max'])
        box_colors_list.extend(['steelblue', 'firebrick'])
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, fontsize=7)
    ax.set_ylabel('Scale s')
    ax.set_title('(c) Scale Penetration')
    ax.grid(True, alpha=0.2, axis='y')

    # (d) Spatial spread
    ax = axes[1, 0]
    spread_data = [res['spreads'] for res in results_list]
    labels = [f'|W|>={res["winding_target"]}' for res in results_list]
    bp = ax.boxplot(spread_data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, c in zip(bp['boxes'], colors[:len(results_list)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Max Chebyshev radius')
    ax.set_title('(d) Spatial Spread')
    ax.grid(True, alpha=0.2, axis='y')

    # (e) Ring fraction
    ax = axes[1, 1]
    ring_data = [res['ring_fracs'] for res in results_list]
    bp = ax.boxplot(ring_data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, c in zip(bp['boxes'], colors[:len(results_list)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Fraction of steps on ring')
    ax.set_title('(e) Ring Occupation')
    ax.grid(True, alpha=0.2, axis='y')

    # (f) Hitting time scaling
    ax = axes[1, 2]
    tgts = [res['winding_target'] for res in results_list]
    means = [np.mean(res['hitting_times']) for res in results_list]
    medians = [np.median(res['hitting_times']) for res in results_list]
    ax.plot(tgts, means, 'o-', markersize=8, color='steelblue', linewidth=2, label='Mean')
    ax.plot(tgts, medians, 's-', markersize=8, color='firebrick', linewidth=2, label='Median')
    if len(tgts) >= 2:
        t_arr = np.array(tgts, dtype=float)
        quad_ref = means[0] * (t_arr / tgts[0])**2
        ax.plot(tgts, quad_ref, '--', color='gray', alpha=0.6, label='W² reference')
    ax.set_xlabel('Winding target |W|')
    ax.set_ylabel('Hitting time (steps)')
    ax.set_title('(f) Hitting Time Scaling')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    target_str = '_'.join(str(t) for t in tgts)
    fname = f'{prefix}_comparison_w{target_str}_opt{opt_label}'
    path = OUTPUT_DIR / f'{fname}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    t0_global = time.time()

    # Configuration
    n_trials = 5000
    max_target = 3        # collect first-passage for |W|=1,2,3
    max_steps = 5_000_000  # generous cap
    options = [('A', True), ('B', False)]

    all_results = {}

    for opt_label, option_a in options:
        print(f'\n{"#"*70}')
        print(f'# Option {opt_label}: {n_trials} trials, max_target={max_target}, '
              f'max_steps={max_steps:,}')
        print(f'{"#"*70}')
        t0 = time.time()

        results = run_batch(n_trials, max_target, max_steps, option_a, base_seed=42)
        elapsed = time.time() - t0
        print(f'\n  Option {opt_label} elapsed: {elapsed:.1f}s')

        for wt in range(1, max_target + 1):
            res = results[wt]
            print_summary(res)
            save_json_summary(res, f'w{wt}_opt{opt_label}')
            plot_single_target(res, prefix='winding_stop')
            all_results[(wt, opt_label)] = res

        # Comparison plot for this option
        plot_comparison(results, opt_label)

    # Cross-comparison table
    print(f'\n\n{"="*90}')
    print('CROSS-COMPARISON: Winding Target x Option')
    print(f'{"="*90}')
    print(f'{"Target":>8} {"Opt":>4} {"Done":>6} {"Rate%":>7} {"Med T":>10} '
          f'{"Mean T":>10} {"Med s_min":>10} {"Med s_max":>10} '
          f'{"Med spread":>11} {"Ring%":>8}')
    print(f'{"-"*90}')
    for wt in range(1, max_target + 1):
        for opt_label, _ in options:
            res = all_results.get((wt, opt_label))
            if res and res['n_completed'] > 0:
                print(f'{wt:>8} {opt_label:>4} {res["n_completed"]:>6} '
                      f'{100*res["n_completed"]/res["n_trials"]:>7.1f} '
                      f'{np.median(res["hitting_times"]):>10.0f} '
                      f'{np.mean(res["hitting_times"]):>10.0f} '
                      f'{np.median(res["s_mins"]):>10.0f} '
                      f'{np.median(res["s_maxs"]):>10.0f} '
                      f'{np.median(res["spreads"]):>11.0f} '
                      f'{np.mean(res["ring_fracs"]):>8.4f}')
            elif res:
                print(f'{wt:>8} {opt_label:>4} {res["n_completed"]:>6} '
                      f'{100*res["n_completed"]/res["n_trials"]:>7.1f} '
                      f'{"---":>10} {"---":>10} {"---":>10} {"---":>10} '
                      f'{"---":>11} {"---":>8}')
    print(f'{"="*90}')

    total_time = time.time() - t0_global
    print(f'\nTotal elapsed: {total_time:.1f}s')
    print('Done!')


if __name__ == '__main__':
    main()
