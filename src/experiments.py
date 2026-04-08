"""
Experiment runners for all measurements specified in the project spec.
Each function runs one experiment and returns results as numpy arrays / dicts.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .walker import Walker, StroboscopicWalker
from .graph import ring_angle_index, is_ring

ESCAPE_THRESHOLD = 10**6


def _make_rng(seed=None):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 3.1 Basic Walk Statistics (No Stroboscope)
# ---------------------------------------------------------------------------

def exp_3_1_winding(n_steps: int, n_trials: int, option: str = 'A',
                    seed: int = 42) -> Dict:
    """3.1(a): Winding number distribution after N steps."""
    rng = _make_rng(seed)
    windings = np.zeros(n_trials)

    for i in range(n_trials):
        w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
        w.run(n_steps)
        windings[i] = w.winding

    return {
        'n_steps': n_steps,
        'n_trials': n_trials,
        'option': option,
        'windings': windings,
        'mean': np.mean(windings),
        'var': np.var(windings),
        'std': np.std(windings),
        'normalized': windings / np.sqrt(n_steps),  # W(N)/sqrt(N)
    }


def exp_3_1_scale_extremes(n_steps: int, n_trials: int, option: str = 'A',
                           seed: int = 42) -> Dict:
    """3.1(b): Max and min scale visited."""
    rng = _make_rng(seed)
    s_mins = np.zeros(n_trials, dtype=int)
    s_maxs = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
        w.run(n_steps)
        s_mins[i] = w.s_min
        s_maxs[i] = w.s_max

    return {
        'n_steps': n_steps,
        'n_trials': n_trials,
        'option': option,
        's_mins': s_mins,
        's_maxs': s_maxs,
        'E_smin': np.mean(s_mins),
        'E_smax': np.mean(s_maxs),
    }


def exp_3_1_scale_profile(n_steps: int, n_trials: int, option: str = 'A',
                          seed: int = 42) -> Dict:
    """3.1(c): Fraction of time at each scale."""
    rng = _make_rng(seed)
    # Aggregate scale visits across trials
    all_profiles = {}

    for i in range(n_trials):
        w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
        w.run(n_steps)
        for s, count in w.scale_visits.items():
            frac = count / n_steps
            if s not in all_profiles:
                all_profiles[s] = []
            all_profiles[s].append(frac)

    # Compute mean fraction at each scale
    scales = sorted(all_profiles.keys())
    mean_fracs = {s: np.mean(all_profiles[s]) for s in scales}

    return {
        'n_steps': n_steps,
        'n_trials': n_trials,
        'option': option,
        'scales': scales,
        'mean_fractions': mean_fracs,
    }


def exp_3_1_return_times(n_steps: int, n_trials: int, option: str = 'A',
                         seed: int = 42) -> Dict:
    """3.1(d): First return time to starting vertex (1,0,0)."""
    rng = _make_rng(seed)
    return_times = []
    n_no_return = 0

    for i in range(n_trials):
        w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
        found = False
        for step in range(1, n_steps + 1):
            w.step()
            if w.x == 1 and w.y == 0 and w.s == 0:
                return_times.append(step)
                found = True
                break
        if not found:
            n_no_return += 1

    return {
        'n_steps': n_steps,
        'n_trials': n_trials,
        'option': option,
        'return_times': np.array(return_times),
        'n_no_return': n_no_return,
        'fraction_returned': len(return_times) / n_trials,
        'mean_return_time': np.mean(return_times) if return_times else np.inf,
        'median_return_time': np.median(return_times) if return_times else np.inf,
    }


# ---------------------------------------------------------------------------
# 3.2 Stroboscopic Walk Statistics
# ---------------------------------------------------------------------------

def exp_3_2_winding(n_outer_ticks: int, n_trials: int, option: str = 'A',
                    adaptive: bool = True, seed: int = 42) -> Dict:
    """3.2(a): Effective winding in outer ticks."""
    rng = _make_rng(seed)
    windings = np.zeros(n_trials)

    for i in range(n_trials):
        sw = StroboscopicWalker(1, 0, 0, option=option, adaptive=adaptive,
                                rng=_make_rng(rng.integers(2**62)))
        sw.run(n_outer_ticks)
        windings[i] = sw.inner.winding

    return {
        'n_outer_ticks': n_outer_ticks,
        'n_trials': n_trials,
        'option': option,
        'adaptive': adaptive,
        'windings': windings,
        'mean': np.mean(windings),
        'var': np.var(windings),
    }


def exp_3_2_scale_penetration(n_outer_ticks: int, n_trials: int, option: str = 'A',
                              adaptive: bool = True, seed: int = 42) -> Dict:
    """3.2(b): Scale penetration in outer ticks."""
    rng = _make_rng(seed)
    s_mins = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        sw = StroboscopicWalker(1, 0, 0, option=option, adaptive=adaptive,
                                rng=_make_rng(rng.integers(2**62)))
        sw.run(n_outer_ticks)
        s_mins[i] = sw.inner.s_min

    return {
        'n_outer_ticks': n_outer_ticks,
        'n_trials': n_trials,
        'option': option,
        'adaptive': adaptive,
        's_mins': s_mins,
        'E_smin': np.mean(s_mins),
    }


def exp_3_2_thermal_blowout(n_outer_ticks: int, n_trials: int, option: str = 'A',
                            adaptive: bool = True, seed: int = 42) -> Dict:
    """3.2(c): Fraction of time on ring vs bulk at deep scales."""
    rng = _make_rng(seed)
    # Track: for walkers reaching s <= -8, what fraction of outer ticks
    # they spend in bulk (not on ring) at that scale
    deep_bulk_fractions = []

    for i in range(n_trials):
        sw = StroboscopicWalker(1, 0, 0, option=option, adaptive=adaptive,
                                rng=_make_rng(rng.integers(2**62)))
        ring_ticks_deep = 0
        bulk_ticks_deep = 0
        reached_deep = False

        for _ in range(n_outer_ticks):
            sw.outer_step()
            if sw.inner.s <= -8:
                reached_deep = True
                if is_ring(sw.inner.x, sw.inner.y):
                    ring_ticks_deep += 1
                else:
                    bulk_ticks_deep += 1

        if reached_deep:
            total = ring_ticks_deep + bulk_ticks_deep
            deep_bulk_fractions.append(bulk_ticks_deep / total if total > 0 else 0)

    return {
        'n_outer_ticks': n_outer_ticks,
        'n_trials': n_trials,
        'option': option,
        'adaptive': adaptive,
        'n_reached_deep': len(deep_bulk_fractions),
        'deep_bulk_fractions': np.array(deep_bulk_fractions),
        'mean_bulk_frac': np.mean(deep_bulk_fractions) if deep_bulk_fractions else None,
    }


# ---------------------------------------------------------------------------
# 3.3 The Aliasing Transition
# ---------------------------------------------------------------------------

def exp_3_3_angular_autocorrelation(n_outer_ticks: int, n_trials: int,
                                    scales: List[int] = None,
                                    option: str = 'A', seed: int = 42) -> Dict:
    """3.3(a): Angular autocorrelation at various depths."""
    if scales is None:
        scales = list(range(-1, -16, -1))

    rng = _make_rng(seed)
    # For each trial, we need to force the walker to a specific scale and observe
    # angular position across outer ticks.
    # Simpler approach: run walk, record (angle, scale) at each outer tick,
    # then bin by scale and compute autocorrelation within each bin.

    # Collect angle sequences per scale
    angle_pairs = {s: [] for s in scales}  # (angle_t, angle_{t+1}) pairs

    for i in range(n_trials):
        sw = StroboscopicWalker(1, 0, 0, option=option, adaptive=False,
                                rng=_make_rng(rng.integers(2**62)))
        prev_angle = ring_angle_index(sw.inner.x, sw.inner.y)
        prev_scale = sw.inner.s

        for _ in range(n_outer_ticks):
            sw.outer_step()
            cur_angle = ring_angle_index(sw.inner.x, sw.inner.y)
            cur_scale = sw.inner.s

            if prev_scale in scales and cur_scale == prev_scale and prev_angle >= 0 and cur_angle >= 0:
                angle_pairs[prev_scale].append((prev_angle, cur_angle))

            prev_angle = cur_angle
            prev_scale = cur_scale

    # Compute autocorrelation: correlation of cos(2*pi*angle/8)
    autocorrs = {}
    for s in scales:
        pairs = angle_pairs[s]
        if len(pairs) < 10:
            autocorrs[s] = np.nan
            continue
        pairs = np.array(pairs)
        a1 = np.cos(2 * np.pi * pairs[:, 0] / 8)
        a2 = np.cos(2 * np.pi * pairs[:, 1] / 8)
        if np.std(a1) < 1e-10 or np.std(a2) < 1e-10:
            autocorrs[s] = np.nan
        else:
            autocorrs[s] = np.corrcoef(a1, a2)[0, 1]

    return {
        'n_outer_ticks': n_outer_ticks,
        'n_trials': n_trials,
        'option': option,
        'scales': scales,
        'autocorrelations': autocorrs,
        'n_pairs': {s: len(angle_pairs[s]) for s in scales},
    }


def exp_3_3_winding_rate(n_steps: int, n_trials: int, option: str = 'A',
                         seed: int = 42) -> Dict:
    """3.3(b): Average winding per step as a function of scale."""
    rng = _make_rng(seed)
    # Track winding contributions at each scale
    scale_winding = {}  # s -> list of per-step winding contributions
    scale_steps = {}    # s -> count of steps at that scale

    for i in range(n_trials):
        w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
        prev_winding = 0.0

        for _ in range(n_steps):
            cur_s = w.s
            w.step()
            dw = w.winding - prev_winding
            prev_winding = w.winding

            if cur_s not in scale_winding:
                scale_winding[cur_s] = 0.0
                scale_steps[cur_s] = 0
            scale_winding[cur_s] += abs(dw)
            scale_steps[cur_s] += 1

    # Compute average winding rate at each scale
    scales = sorted(scale_winding.keys())
    winding_rates = {}
    for s in scales:
        if scale_steps[s] > 0:
            winding_rates[s] = scale_winding[s] / scale_steps[s]

    return {
        'n_steps': n_steps,
        'n_trials': n_trials,
        'option': option,
        'scales': scales,
        'winding_rates': winding_rates,
        'scale_steps': scale_steps,
    }


# ---------------------------------------------------------------------------
# 3.4 Comparison with Hyperbolic Random Walk
# ---------------------------------------------------------------------------

def exp_3_4_radial_drift(n_steps_list: List[int], n_trials: int, option: str = 'A',
                         seed: int = 42) -> Dict:
    """3.4(a): E[r(N)] where r = max(|x|, |y|)."""
    rng = _make_rng(seed)
    results = {}

    for n_steps in n_steps_list:
        radii = np.zeros(n_trials)
        for i in range(n_trials):
            w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
            w.run(n_steps)
            radii[i] = max(abs(w.x), abs(w.y))
        results[n_steps] = {
            'mean_r': np.mean(radii),
            'std_r': np.std(radii),
            'radii': radii,
        }

    return {
        'n_steps_list': n_steps_list,
        'n_trials': n_trials,
        'option': option,
        'results': results,
    }


def exp_3_4_scale_drift(n_steps_list: List[int], n_trials: int, option: str = 'A',
                        seed: int = 42) -> Dict:
    """3.4(b): E[s(N)] and Var[s(N)]."""
    rng = _make_rng(seed)
    results = {}

    for n_steps in n_steps_list:
        scales = np.zeros(n_trials)
        for i in range(n_trials):
            w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
            w.run(n_steps)
            scales[i] = w.s
        results[n_steps] = {
            'mean_s': np.mean(scales),
            'var_s': np.var(scales),
            'scales': scales,
        }

    return {
        'n_steps_list': n_steps_list,
        'n_trials': n_trials,
        'option': option,
        'results': results,
    }


def exp_3_4_spectral_dimension(n_steps_list: List[int], n_trials: int,
                               option: str = 'A', seed: int = 42) -> Dict:
    """3.4(c): Return probability P(N) to estimate spectral dimension."""
    rng = _make_rng(seed)
    # P(N) = probability of being at start after N steps
    results = {}

    for n_steps in n_steps_list:
        n_returned = 0
        for i in range(n_trials):
            w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))
            w.run(n_steps)
            if w.x == 1 and w.y == 0 and w.s == 0:
                n_returned += 1
        results[n_steps] = {
            'return_prob': n_returned / n_trials,
            'n_returned': n_returned,
        }

    return {
        'n_steps_list': n_steps_list,
        'n_trials': n_trials,
        'option': option,
        'results': results,
    }


# ---------------------------------------------------------------------------
# 4. Propp Pi-Estimation: Excursion Ratios
# ---------------------------------------------------------------------------

def exp_4_excursion_ratios(max_steps: int, n_excursions: int, winding_target: int = 1,
                           option: str = 'A', seed: int = 42) -> Dict:
    """
    4(a,b): Find excursions (start and end at same ring vertex with given winding).
    Compute 8/T for each excursion.
    """
    rng = _make_rng(seed)
    ratios = []
    excursion_lengths = []
    n_timeout = 0

    trial = 0
    while len(ratios) < n_excursions and trial < n_excursions * 100:
        trial += 1
        w = Walker(1, 0, 0, option=option, rng=_make_rng(rng.integers(2**62)))

        for step in range(1, max_steps + 1):
            w.step()
            # Check if we've returned to (1,0,0) with target winding
            if (w.x == 1 and w.y == 0 and w.s == 0 and
                    abs(w.winding - winding_target) < 1e-9):
                ratios.append(8.0 / step)
                excursion_lengths.append(step)
                break
        else:
            n_timeout += 1

    ratios = np.array(ratios)
    excursion_lengths = np.array(excursion_lengths)

    return {
        'max_steps': max_steps,
        'n_excursions_found': len(ratios),
        'n_excursions_target': n_excursions,
        'winding_target': winding_target,
        'option': option,
        'ratios': ratios,
        'excursion_lengths': excursion_lengths,
        'mean_ratio': np.mean(ratios) if len(ratios) > 0 else None,
        'std_ratio': np.std(ratios) if len(ratios) > 0 else None,
        'n_timeout': n_timeout,
        'pi_over_4': np.pi / 4,
        'ln_2': np.log(2),
    }
