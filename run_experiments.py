#!/usr/bin/env python3
"""
Main experiment runner for Random Walks on the Discrete Punctured Plane.

Usage:
    python run_experiments.py [experiment_name] [--quick] [--option A|B|both]

    experiment_name: one of 'all', '3.1', '3.2', '3.3', '3.4', '4', 'quick_check'
    --quick: use small N and trial counts for fast iteration
    --option: which stepping rule to use (default: both)
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from src.experiments import (
    exp_3_1_winding, exp_3_1_scale_extremes, exp_3_1_scale_profile,
    exp_3_1_return_times,
    exp_3_2_winding, exp_3_2_scale_penetration, exp_3_2_thermal_blowout,
    exp_3_3_angular_autocorrelation, exp_3_3_winding_rate,
    exp_3_4_radial_drift, exp_3_4_scale_drift, exp_3_4_spectral_dimension,
    exp_4_excursion_ratios,
)
from src.plotting import (
    plot_winding_distribution, plot_scale_extremes, plot_scale_profile,
    plot_return_times, plot_autocorrelation, plot_winding_rate,
    plot_radial_drift, plot_scale_drift, plot_spectral_dimension,
    plot_excursion_ratios, plot_option_comparison,
)

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)


def save_summary(name: str, results: dict):
    """Save non-array results as JSON summary."""
    summary = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            summary[k] = {'shape': list(v.shape), 'mean': float(np.mean(v)),
                         'std': float(np.std(v)), 'min': float(np.min(v)),
                         'max': float(np.max(v))}
        elif isinstance(v, dict):
            summary[k] = {str(kk): (float(vv) if isinstance(vv, (int, float, np.floating, np.integer))
                                    else str(vv))
                         for kk, vv in v.items()}
        elif isinstance(v, (int, float, np.floating, np.integer)):
            summary[k] = float(v)
        elif isinstance(v, list):
            summary[k] = str(v)[:200]
        elif v is None:
            summary[k] = None
        else:
            summary[k] = str(v)[:200]

    path = OUTPUT_DIR / f'{name}_summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Saved summary: {path}')


def run_3_1(options, quick=False):
    """Section 3.1: Basic Walk Statistics."""
    if quick:
        n_steps_list = [1000, 5000]
        n_trials = 500
    else:
        n_steps_list = [10_000, 100_000, 1_000_000]
        n_trials = 10_000

    for opt in options:
        for n_steps in n_steps_list:
            print(f'\n=== 3.1(a) Winding: N={n_steps}, Option {opt} ===')
            t0 = time.time()
            res = exp_3_1_winding(n_steps, n_trials, option=opt)
            print(f'  Time: {time.time()-t0:.1f}s | Mean={res["mean"]:.4f}, Var={res["var"]:.4f}')
            plot_winding_distribution(res)
            save_summary(f'3_1a_opt{opt}_N{n_steps}', res)

            print(f'\n=== 3.1(b) Scale extremes: N={n_steps}, Option {opt} ===')
            t0 = time.time()
            res = exp_3_1_scale_extremes(n_steps, n_trials, option=opt)
            print(f'  Time: {time.time()-t0:.1f}s | E[smin]={res["E_smin"]:.2f}, E[smax]={res["E_smax"]:.2f}')
            plot_scale_extremes(res)
            save_summary(f'3_1b_opt{opt}_N{n_steps}', res)

            print(f'\n=== 3.1(c) Scale profile: N={n_steps}, Option {opt} ===')
            t0 = time.time()
            res = exp_3_1_scale_profile(n_steps, n_trials, option=opt)
            print(f'  Time: {time.time()-t0:.1f}s')
            plot_scale_profile(res)
            save_summary(f'3_1c_opt{opt}_N{n_steps}', res)

        print(f'\n=== 3.1(d) Return times: Option {opt} ===')
        t0 = time.time()
        res = exp_3_1_return_times(n_steps_list[-1], n_trials, option=opt)
        print(f'  Time: {time.time()-t0:.1f}s | Returned: {res["fraction_returned"]:.3f}, '
              f'Mean: {res["mean_return_time"]:.1f}')
        plot_return_times(res)
        save_summary(f'3_1d_opt{opt}', res)


def run_3_2(options, quick=False):
    """Section 3.2: Stroboscopic Walk Statistics."""
    if quick:
        n_ticks = 1000
        n_trials = 500
    else:
        n_ticks = 100_000
        n_trials = 5_000

    for opt in options:
        for adaptive in [True, False]:
            mode = 'adaptive' if adaptive else 'fixed'
            print(f'\n=== 3.2(a) Stroboscopic winding: Option {opt}, {mode} ===')
            t0 = time.time()
            res = exp_3_2_winding(n_ticks, n_trials, option=opt, adaptive=adaptive)
            print(f'  Time: {time.time()-t0:.1f}s | Mean={res["mean"]:.4f}, Var={res["var"]:.4f}')
            save_summary(f'3_2a_opt{opt}_{mode}', res)

            print(f'\n=== 3.2(b) Scale penetration: Option {opt}, {mode} ===')
            t0 = time.time()
            res = exp_3_2_scale_penetration(n_ticks, n_trials, option=opt, adaptive=adaptive)
            print(f'  Time: {time.time()-t0:.1f}s | E[smin]={res["E_smin"]:.2f}')
            save_summary(f'3_2b_opt{opt}_{mode}', res)

            print(f'\n=== 3.2(c) Thermal blowout: Option {opt}, {mode} ===')
            t0 = time.time()
            res = exp_3_2_thermal_blowout(n_ticks, min(n_trials, 1000), option=opt, adaptive=adaptive)
            print(f'  Time: {time.time()-t0:.1f}s | Reached deep: {res["n_reached_deep"]}, '
                  f'Mean bulk frac: {res["mean_bulk_frac"]}')
            save_summary(f'3_2c_opt{opt}_{mode}', res)


def run_3_3(options, quick=False):
    """Section 3.3: The Aliasing Transition."""
    if quick:
        n_steps = 5000
        n_ticks = 1000
        n_trials = 200
    else:
        n_steps = 100_000
        n_ticks = 50_000
        n_trials = 2_000

    for opt in options:
        print(f'\n=== 3.3(a) Angular autocorrelation: Option {opt} ===')
        t0 = time.time()
        res = exp_3_3_angular_autocorrelation(n_ticks, n_trials, option=opt)
        print(f'  Time: {time.time()-t0:.1f}s')
        for s in sorted(res['autocorrelations'].keys()):
            ac = res['autocorrelations'][s]
            n_p = res['n_pairs'][s]
            if not np.isnan(ac):
                print(f'  s={s:3d}: autocorr={ac:.4f} (n={n_p})')
        plot_autocorrelation(res)
        save_summary(f'3_3a_opt{opt}', res)

        print(f'\n=== 3.3(b) Winding rate vs scale: Option {opt} ===')
        t0 = time.time()
        res = exp_3_3_winding_rate(n_steps, n_trials, option=opt)
        print(f'  Time: {time.time()-t0:.1f}s')
        for s in sorted(res['winding_rates'].keys()):
            if -20 <= s <= 0:
                print(f'  s={s:3d}: rate={res["winding_rates"][s]:.6f} ({res["scale_steps"][s]} steps)')
        plot_winding_rate(res)
        save_summary(f'3_3b_opt{opt}', res)


def run_3_4(options, quick=False):
    """Section 3.4: Comparison with Hyperbolic Random Walk."""
    if quick:
        n_steps_list = [100, 500, 1000, 5000]
        n_trials = 500
    else:
        n_steps_list = [1000, 5000, 10_000, 50_000, 100_000]
        n_trials = 5_000

    for opt in options:
        print(f'\n=== 3.4(a) Radial drift: Option {opt} ===')
        t0 = time.time()
        res = exp_3_4_radial_drift(n_steps_list, n_trials, option=opt)
        print(f'  Time: {time.time()-t0:.1f}s')
        for n in n_steps_list:
            print(f'  N={n:>7d}: E[r]={res["results"][n]["mean_r"]:.2f}')
        plot_radial_drift(res)
        save_summary(f'3_4a_opt{opt}', res)

        print(f'\n=== 3.4(b) Scale drift: Option {opt} ===')
        t0 = time.time()
        res = exp_3_4_scale_drift(n_steps_list, n_trials, option=opt)
        print(f'  Time: {time.time()-t0:.1f}s')
        for n in n_steps_list:
            r = res['results'][n]
            print(f'  N={n:>7d}: E[s]={r["mean_s"]:.3f}, Var[s]={r["var_s"]:.3f}')
        plot_scale_drift(res)
        save_summary(f'3_4b_opt{opt}', res)

        print(f'\n=== 3.4(c) Spectral dimension: Option {opt} ===')
        t0 = time.time()
        n_trials_spec = min(n_trials, 2000) if quick else 10_000
        res = exp_3_4_spectral_dimension(n_steps_list, n_trials_spec, option=opt)
        print(f'  Time: {time.time()-t0:.1f}s')
        for n in n_steps_list:
            print(f'  N={n:>7d}: P(N)={res["results"][n]["return_prob"]:.6f}')
        plot_spectral_dimension(res)
        save_summary(f'3_4c_opt{opt}', res)


def run_4(options, quick=False):
    """Section 4: Propp Pi-Estimation."""
    if quick:
        max_steps = 50_000
        n_excursions = 200
    else:
        max_steps = 1_000_000
        n_excursions = 5_000

    for opt in options:
        for winding_target in [1, 2]:
            print(f'\n=== 4. Excursion ratios: winding={winding_target}, Option {opt} ===')
            t0 = time.time()
            res = exp_4_excursion_ratios(max_steps, n_excursions,
                                         winding_target=winding_target, option=opt)
            print(f'  Time: {time.time()-t0:.1f}s')
            print(f'  Found: {res["n_excursions_found"]}/{n_excursions}')
            if res['mean_ratio'] is not None:
                print(f'  E[8/T] = {res["mean_ratio"]:.6f}')
                print(f'  π/4    = {res["pi_over_4"]:.6f}')
                print(f'  ln(2)  = {res["ln_2"]:.6f}')
            plot_excursion_ratios(res)
            save_summary(f'4_excursion_w{winding_target}_opt{opt}', res)


def run_quick_check():
    """Fast sanity check: small runs to verify everything works."""
    print('=== Quick sanity check ===')

    from src.walker import Walker
    from src.graph import is_ring, ring_angle_index

    for opt in ['A', 'B']:
        w = Walker(1, 0, 0, option=opt)
        print(f'\nOption {opt}: neighbors of (1,0,0) = {w.get_neighbors()}')

        w.run(100)
        print(f'  After 100 steps: ({w.x}, {w.y}, {w.s}), winding={w.winding:.3f}, '
              f'smin={w.s_min}, smax={w.s_max}')

    print('\nRunning mini experiments...')
    res = exp_3_1_winding(1000, 100, option='A')
    print(f'  Winding (N=1000, 100 trials): mean={res["mean"]:.4f}, var={res["var"]:.4f}')
    plot_winding_distribution(res)

    res = exp_3_1_return_times(10000, 1000, option='A')
    print(f'  Return times: {res["fraction_returned"]:.3f} returned, '
          f'mean={res["mean_return_time"]:.1f}')
    plot_return_times(res)

    res = exp_4_excursion_ratios(50000, 50, winding_target=1, option='A')
    print(f'  Excursion ratios (w=1): found {res["n_excursions_found"]}, '
          f'mean={res["mean_ratio"]}')
    if res['mean_ratio'] is not None:
        print(f'    E[8/T]={res["mean_ratio"]:.6f}, π/4={np.pi/4:.6f}, ln2={np.log(2):.6f}')

    print('\nQuick check done.')


def main():
    parser = argparse.ArgumentParser(description='Punctured Plane Walk Experiments')
    parser.add_argument('experiment', nargs='?', default='quick_check',
                       choices=['all', '3.1', '3.2', '3.3', '3.4', '4', 'quick_check'],
                       help='Which experiment to run')
    parser.add_argument('--quick', action='store_true',
                       help='Use small parameters for fast iteration')
    parser.add_argument('--option', choices=['A', 'B', 'both'], default='both',
                       help='Stepping rule option')
    args = parser.parse_args()

    if args.option == 'both':
        options = ['A', 'B']
    else:
        options = [args.option]

    if args.experiment == 'quick_check':
        run_quick_check()
    elif args.experiment == '3.1':
        run_3_1(options, args.quick)
    elif args.experiment == '3.2':
        run_3_2(options, args.quick)
    elif args.experiment == '3.3':
        run_3_3(options, args.quick)
    elif args.experiment == '3.4':
        run_3_4(options, args.quick)
    elif args.experiment == '4':
        run_4(options, args.quick)
    elif args.experiment == 'all':
        run_3_1(options, args.quick)
        run_3_2(options, args.quick)
        run_3_3(options, args.quick)
        run_3_4(options, args.quick)
        run_4(options, args.quick)


if __name__ == '__main__':
    main()
