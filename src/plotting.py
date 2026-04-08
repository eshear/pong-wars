"""
Plotting utilities for all experiments.
Produces the deliverables: histograms, log-log plots, comparison plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def _save(fig, name: str):
    path = OUTPUT_DIR / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_winding_distribution(results: Dict, prefix: str = '3_1a'):
    """Plot winding number histogram and normalized distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    w = results['windings']
    n = results['n_steps']
    opt = results['option']

    axes[0].hist(w, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Winding number W(N)')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Winding distribution (N={n}, Option {opt})')
    axes[0].axvline(results['mean'], color='red', linestyle='--', label=f'mean={results["mean"]:.3f}')
    axes[0].legend()

    # Normalized W/sqrt(N) vs Gaussian
    wn = results['normalized']
    axes[1].hist(wn, bins=50, density=True, alpha=0.7, edgecolor='black', label='W(N)/√N')
    x = np.linspace(wn.min(), wn.max(), 200)
    sigma = np.std(wn)
    if sigma > 0:
        gaussian = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        axes[1].plot(x, gaussian, 'r-', label=f'Gaussian (σ={sigma:.3f})')
    axes[1].set_xlabel('W(N)/√N')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Normalized winding (N={n}, Option {opt})')
    axes[1].legend()

    fig.tight_layout()
    _save(fig, f'{prefix}_winding_opt{opt}_N{n}')


def plot_scale_extremes(results: Dict, prefix: str = '3_1b'):
    """Plot distributions of smin and smax."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    opt = results['option']
    n = results['n_steps']

    axes[0].hist(results['s_mins'], bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Minimum scale smin')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'smin distribution (N={n}, Option {opt})\nE[smin]={results["E_smin"]:.2f}')

    axes[1].hist(results['s_maxs'], bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Maximum scale smax')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'smax distribution (N={n}, Option {opt})\nE[smax]={results["E_smax"]:.2f}')

    fig.tight_layout()
    _save(fig, f'{prefix}_scale_extremes_opt{opt}_N{n}')


def plot_scale_profile(results: Dict, prefix: str = '3_1c'):
    """Plot fraction of time at each scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scales = results['scales']
    fracs = [results['mean_fractions'][s] for s in scales]
    opt = results['option']
    n = results['n_steps']

    ax.bar(scales, fracs, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Scale s')
    ax.set_ylabel('Mean fraction of time')
    ax.set_title(f'Scale profile (N={n}, Option {opt})')
    ax.set_yscale('log')

    _save(fig, f'{prefix}_scale_profile_opt{opt}_N{n}')


def plot_return_times(results: Dict, prefix: str = '3_1d'):
    """Plot return time distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rt = results['return_times']
    opt = results['option']
    n = results['n_steps']

    if len(rt) > 0:
        axes[0].hist(rt, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('First return time')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Return times (N={n}, Option {opt})\n'
                         f'Returned: {results["fraction_returned"]:.3f}, '
                         f'Mean: {results["mean_return_time"]:.1f}')

        # Log-log of survival function
        sorted_rt = np.sort(rt)
        survival = 1 - np.arange(1, len(sorted_rt) + 1) / len(sorted_rt)
        axes[1].loglog(sorted_rt, survival, '.', markersize=2)
        axes[1].set_xlabel('Return time t')
        axes[1].set_ylabel('P(T > t)')
        axes[1].set_title(f'Return time survival (log-log)')
    else:
        axes[0].text(0.5, 0.5, 'No returns observed', ha='center', va='center',
                    transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'No returns observed', ha='center', va='center',
                    transform=axes[1].transAxes)

    fig.tight_layout()
    _save(fig, f'{prefix}_return_times_opt{opt}_N{n}')


def plot_autocorrelation(results: Dict, prefix: str = '3_3a'):
    """Plot angular autocorrelation vs scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scales = results['scales']
    autocorrs = results['autocorrelations']
    opt = results['option']

    s_vals = [s for s in scales if not np.isnan(autocorrs.get(s, np.nan))]
    a_vals = [autocorrs[s] for s in s_vals]

    ax.plot(s_vals, a_vals, 'o-', markersize=6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-8, color='red', linestyle='--', alpha=0.5, label='s = -8 (aliasing)')
    ax.set_xlabel('Scale s')
    ax.set_ylabel('Angular autocorrelation')
    ax.set_title(f'Angular autocorrelation vs depth (Option {opt})')
    ax.legend()

    _save(fig, f'{prefix}_autocorr_opt{opt}')


def plot_winding_rate(results: Dict, prefix: str = '3_3b'):
    """Plot winding rate vs scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    wr = results['winding_rates']
    opt = results['option']

    s_vals = sorted(s for s in wr if -20 <= s <= 0)
    rates = [wr[s] for s in s_vals]

    ax.plot(s_vals, rates, 'o-', markersize=6)
    ax.axvline(-8, color='red', linestyle='--', alpha=0.5, label='s = -8 (aliasing)')
    ax.set_xlabel('Scale s')
    ax.set_ylabel('Mean |winding| per step')
    ax.set_title(f'Winding rate vs scale (Option {opt})')
    ax.legend()

    _save(fig, f'{prefix}_winding_rate_opt{opt}')


def plot_radial_drift(results: Dict, prefix: str = '3_4a'):
    """Log-log plot of E[r(N)] vs N."""
    fig, ax = plt.subplots(figsize=(10, 6))

    opt = results['option']
    n_list = results['n_steps_list']
    mean_rs = [results['results'][n]['mean_r'] for n in n_list]

    ax.loglog(n_list, mean_rs, 'o-', markersize=8, label='E[r(N)]')

    # Reference lines
    n_arr = np.array(n_list, dtype=float)
    ax.loglog(n_list, np.sqrt(n_arr) * mean_rs[0] / np.sqrt(n_list[0]),
              '--', alpha=0.5, label='√N scaling')
    ax.loglog(n_list, n_arr * mean_rs[0] / n_list[0],
              '--', alpha=0.5, label='N scaling')

    ax.set_xlabel('N (steps)')
    ax.set_ylabel('E[r(N)]')
    ax.set_title(f'Radial drift (Option {opt})')
    ax.legend()

    _save(fig, f'{prefix}_radial_drift_opt{opt}')


def plot_scale_drift(results: Dict, prefix: str = '3_4b'):
    """Plot E[s(N)] and Var[s(N)] vs N."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    opt = results['option']
    n_list = results['n_steps_list']
    mean_s = [results['results'][n]['mean_s'] for n in n_list]
    var_s = [results['results'][n]['var_s'] for n in n_list]

    axes[0].semilogx(n_list, mean_s, 'o-', markersize=8)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('N (steps)')
    axes[0].set_ylabel('E[s(N)]')
    axes[0].set_title(f'Scale drift (Option {opt})')

    axes[1].loglog(n_list, var_s, 'o-', markersize=8, label='Var[s(N)]')
    n_arr = np.array(n_list, dtype=float)
    axes[1].loglog(n_list, n_arr * var_s[0] / n_list[0], '--', alpha=0.5, label='N scaling')
    axes[1].set_xlabel('N (steps)')
    axes[1].set_ylabel('Var[s(N)]')
    axes[1].set_title(f'Scale variance (Option {opt})')
    axes[1].legend()

    fig.tight_layout()
    _save(fig, f'{prefix}_scale_drift_opt{opt}')


def plot_spectral_dimension(results: Dict, prefix: str = '3_4c'):
    """Log-log plot of P(N) vs N to estimate spectral dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))

    opt = results['option']
    n_list = results['n_steps_list']
    probs = [results['results'][n]['return_prob'] for n in n_list]

    # Filter out zeros for log-log
    valid = [(n, p) for n, p in zip(n_list, probs) if p > 0]
    if valid:
        ns, ps = zip(*valid)
        ax.loglog(ns, ps, 'o-', markersize=8, label='P(N)')

        # Reference lines for ds=2 (P ~ N^-1) and ds=1 (P ~ N^-0.5)
        n_arr = np.array(ns, dtype=float)
        ax.loglog(ns, ps[0] * (n_arr[0] / n_arr), '--', alpha=0.5, label='N^-1 (ds=2)')
        ax.loglog(ns, ps[0] * np.sqrt(n_arr[0] / n_arr), '--', alpha=0.5, label='N^-0.5 (ds=1)')

    ax.set_xlabel('N (steps)')
    ax.set_ylabel('P(N)')
    ax.set_title(f'Return probability / Spectral dimension (Option {opt})')
    ax.legend()

    _save(fig, f'{prefix}_spectral_dim_opt{opt}')


def plot_excursion_ratios(results: Dict, prefix: str = '4'):
    """Plot excursion ratio distribution and compare to pi/4, ln(2)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ratios = results['ratios']
    wt = results['winding_target']
    opt = results['option']

    if len(ratios) > 0:
        ax.hist(ratios, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(results['mean_ratio'], color='blue', linestyle='--',
                  label=f'Mean = {results["mean_ratio"]:.6f}')
        ax.axvline(np.pi / 4, color='red', linestyle='--',
                  label=f'π/4 = {np.pi/4:.6f}')
        ax.axvline(np.log(2), color='green', linestyle='--',
                  label=f'ln(2) = {np.log(2):.6f}')
        ax.set_xlabel('8/T (excursion ratio)')
        ax.set_ylabel('Density')
        ax.set_title(f'Excursion ratios (winding={wt}, Option {opt})\n'
                    f'n={results["n_excursions_found"]}, mean={results["mean_ratio"]:.6f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No excursions found', ha='center', va='center',
               transform=ax.transAxes)

    _save(fig, f'{prefix}_excursion_ratio_w{wt}_opt{opt}')


def plot_option_comparison(results_a: Dict, results_b: Dict, quantity: str,
                          prefix: str = 'comparison'):
    """Generic comparison plot between Option A and Option B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (res, label) in zip(axes, [(results_a, 'Option A'), (results_b, 'Option B')]):
        if quantity == 'winding':
            ax.hist(res['windings'], bins=50, density=True, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Winding')
            ax.set_title(f'{label}: mean={res["mean"]:.3f}, var={res["var"]:.3f}')
        elif quantity == 'scale_extremes':
            ax.hist(res['s_mins'], bins=30, density=True, alpha=0.7, label='smin')
            ax.hist(res['s_maxs'], bins=30, density=True, alpha=0.5, label='smax')
            ax.set_title(f'{label}: E[smin]={res["E_smin"]:.2f}, E[smax]={res["E_smax"]:.2f}')
            ax.legend()
        ax.set_ylabel('Density')

    fig.suptitle(f'{quantity} comparison: Option A vs Option B')
    fig.tight_layout()
    _save(fig, f'{prefix}_{quantity}')
