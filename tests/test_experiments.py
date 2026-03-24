"""Tests for src/experiments.py — experiment runners produce well-formed results."""

import numpy as np
import pytest
from src.experiments import (
    exp_3_1_winding,
    exp_3_1_scale_extremes,
    exp_3_1_scale_profile,
    exp_3_1_return_times,
    exp_3_2_winding,
    exp_3_2_scale_penetration,
    exp_3_2_thermal_blowout,
    exp_3_3_angular_autocorrelation,
    exp_3_3_winding_rate,
    exp_3_4_radial_drift,
    exp_3_4_scale_drift,
    exp_3_4_spectral_dimension,
    exp_4_excursion_ratios,
)

# Use small parameters for fast tests
SMALL_STEPS = 200
SMALL_TRIALS = 10
SMALL_SEED = 42


# ---------------------------------------------------------------------------
# 3.1 Basic Walk Statistics
# ---------------------------------------------------------------------------

class TestExp31Winding:
    @pytest.fixture(params=['A', 'B'])
    def result(self, request):
        return exp_3_1_winding(SMALL_STEPS, SMALL_TRIALS, option=request.param, seed=SMALL_SEED)

    def test_keys(self, result):
        for key in ['n_steps', 'n_trials', 'option', 'windings', 'mean', 'var', 'std', 'normalized']:
            assert key in result

    def test_windings_shape(self, result):
        assert result['windings'].shape == (SMALL_TRIALS,)

    def test_normalized_shape(self, result):
        assert result['normalized'].shape == (SMALL_TRIALS,)

    def test_std_nonnegative(self, result):
        assert result['std'] >= 0

    def test_var_nonnegative(self, result):
        assert result['var'] >= 0

    def test_reproducible(self):
        r1 = exp_3_1_winding(100, 5, seed=99)
        r2 = exp_3_1_winding(100, 5, seed=99)
        np.testing.assert_array_equal(r1['windings'], r2['windings'])


class TestExp31ScaleExtremes:
    @pytest.fixture(params=['A', 'B'])
    def result(self, request):
        return exp_3_1_scale_extremes(SMALL_STEPS, SMALL_TRIALS, option=request.param, seed=SMALL_SEED)

    def test_keys(self, result):
        for key in ['n_steps', 'n_trials', 'option', 's_mins', 's_maxs', 'E_smin', 'E_smax']:
            assert key in result

    def test_shapes(self, result):
        assert result['s_mins'].shape == (SMALL_TRIALS,)
        assert result['s_maxs'].shape == (SMALL_TRIALS,)

    def test_smin_le_smax(self, result):
        assert np.all(result['s_mins'] <= result['s_maxs'])

    def test_smin_le_zero(self, result):
        """Started at s=0, so s_min <= 0 and s_max >= 0."""
        assert np.all(result['s_mins'] <= 0)
        assert np.all(result['s_maxs'] >= 0)


class TestExp31ScaleProfile:
    @pytest.fixture(params=['A', 'B'])
    def result(self, request):
        return exp_3_1_scale_profile(SMALL_STEPS, SMALL_TRIALS, option=request.param, seed=SMALL_SEED)

    def test_keys(self, result):
        for key in ['n_steps', 'n_trials', 'option', 'scales', 'mean_fractions']:
            assert key in result

    def test_fractions_positive(self, result):
        for s, f in result['mean_fractions'].items():
            assert f >= 0


class TestExp31ReturnTimes:
    @pytest.fixture(params=['A', 'B'])
    def result(self, request):
        return exp_3_1_return_times(SMALL_STEPS, SMALL_TRIALS, option=request.param, seed=SMALL_SEED)

    def test_keys(self, result):
        for key in ['n_steps', 'n_trials', 'option', 'return_times', 'n_no_return',
                     'fraction_returned', 'mean_return_time', 'median_return_time']:
            assert key in result

    def test_fraction_bounded(self, result):
        assert 0 <= result['fraction_returned'] <= 1

    def test_return_times_positive(self, result):
        if len(result['return_times']) > 0:
            assert np.all(result['return_times'] > 0)
            assert np.all(result['return_times'] <= SMALL_STEPS)

    def test_counts_add_up(self, result):
        assert len(result['return_times']) + result['n_no_return'] == SMALL_TRIALS


# ---------------------------------------------------------------------------
# 3.2 Stroboscopic Walk Statistics
# ---------------------------------------------------------------------------

class TestExp32Winding:
    def test_basic(self):
        result = exp_3_2_winding(50, SMALL_TRIALS, option='A', seed=SMALL_SEED)
        assert result['windings'].shape == (SMALL_TRIALS,)
        assert 'mean' in result and 'var' in result


class TestExp32ScalePenetration:
    def test_basic(self):
        result = exp_3_2_scale_penetration(50, SMALL_TRIALS, option='A', seed=SMALL_SEED)
        assert result['s_mins'].shape == (SMALL_TRIALS,)
        assert np.all(result['s_mins'] <= 0)


class TestExp32ThermalBlowout:
    def test_basic(self):
        result = exp_3_2_thermal_blowout(50, SMALL_TRIALS, option='A', seed=SMALL_SEED)
        assert 'n_reached_deep' in result
        assert result['n_reached_deep'] >= 0
        assert result['n_reached_deep'] <= SMALL_TRIALS


# ---------------------------------------------------------------------------
# 3.3 Aliasing Transition
# ---------------------------------------------------------------------------

class TestExp33AngularAutocorrelation:
    def test_basic(self):
        result = exp_3_3_angular_autocorrelation(
            50, SMALL_TRIALS, scales=[-1, -2], option='A', seed=SMALL_SEED)
        assert 'autocorrelations' in result
        assert 'n_pairs' in result
        assert set(result['scales']) == {-1, -2}


class TestExp33WindingRate:
    def test_basic(self):
        result = exp_3_3_winding_rate(SMALL_STEPS, SMALL_TRIALS, option='A', seed=SMALL_SEED)
        assert 'winding_rates' in result
        assert 'scale_steps' in result
        # At least scale 0 should have some steps
        assert 0 in result['scale_steps']
        assert result['scale_steps'][0] > 0


# ---------------------------------------------------------------------------
# 3.4 Hyperbolic Comparison
# ---------------------------------------------------------------------------

class TestExp34RadialDrift:
    def test_basic(self):
        result = exp_3_4_radial_drift([50, 100], SMALL_TRIALS, option='A', seed=SMALL_SEED)
        assert 50 in result['results']
        assert 100 in result['results']
        for n in [50, 100]:
            assert result['results'][n]['mean_r'] > 0
            assert result['results'][n]['radii'].shape == (SMALL_TRIALS,)


class TestExp34ScaleDrift:
    def test_basic(self):
        result = exp_3_4_scale_drift([50, 100], SMALL_TRIALS, option='A', seed=SMALL_SEED)
        for n in [50, 100]:
            assert 'mean_s' in result['results'][n]
            assert 'var_s' in result['results'][n]
            assert result['results'][n]['var_s'] >= 0


class TestExp34SpectralDimension:
    def test_basic(self):
        result = exp_3_4_spectral_dimension([10, 20], SMALL_TRIALS, option='A', seed=SMALL_SEED)
        for n in [10, 20]:
            assert 0 <= result['results'][n]['return_prob'] <= 1
            assert result['results'][n]['n_returned'] >= 0


# ---------------------------------------------------------------------------
# 4. Propp Pi-Estimation
# ---------------------------------------------------------------------------

class TestExp4ExcursionRatios:
    def test_basic(self):
        result = exp_4_excursion_ratios(
            max_steps=5000, n_excursions=3, winding_target=1, option='A', seed=SMALL_SEED)
        assert 'ratios' in result
        assert 'excursion_lengths' in result
        assert result['n_excursions_found'] >= 0
        assert result['n_excursions_found'] <= 3
        assert 'pi_over_4' in result
        assert 'ln_2' in result

    def test_ratios_positive(self):
        result = exp_4_excursion_ratios(
            max_steps=5000, n_excursions=3, winding_target=1, option='A', seed=SMALL_SEED)
        if len(result['ratios']) > 0:
            assert np.all(result['ratios'] > 0)

    def test_excursion_lengths_positive(self):
        result = exp_4_excursion_ratios(
            max_steps=5000, n_excursions=3, winding_target=1, option='A', seed=SMALL_SEED)
        if len(result['excursion_lengths']) > 0:
            assert np.all(result['excursion_lengths'] > 0)
            assert np.all(result['excursion_lengths'] <= 5000)


# ---------------------------------------------------------------------------
# Cross-cutting: Option A vs B produce different distributions
# ---------------------------------------------------------------------------

class TestOptionComparison:
    def test_winding_differs_between_options(self):
        """Options A and B should yield different winding distributions (different graphs)."""
        ra = exp_3_1_winding(500, 20, option='A', seed=42)
        rb = exp_3_1_winding(500, 20, option='B', seed=42)
        # Same seed but different graph structure → different windings
        assert not np.array_equal(ra['windings'], rb['windings'])

    def test_scale_extremes_differ_between_options(self):
        ra = exp_3_1_scale_extremes(500, 20, option='A', seed=42)
        rb = exp_3_1_scale_extremes(500, 20, option='B', seed=42)
        # Not identical (different chimney connectivity)
        assert not np.array_equal(ra['s_mins'], rb['s_mins']) or \
               not np.array_equal(ra['s_maxs'], rb['s_maxs'])
