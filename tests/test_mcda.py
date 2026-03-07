"""Tests for solarex.analysis.mcda."""

import numpy as np
import pytest

from solarex.analysis.mcda import compute_mcda_scores, entropy_weights, pca_weights


class TestEntropyWeights:
    def test_uniform_data(self):
        matrix = np.ones((10, 3)) * 0.5
        w = entropy_weights(matrix)
        assert w.shape == (3,)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_varied_data(self):
        rng = np.random.default_rng(42)
        matrix = rng.random((50, 4))
        w = entropy_weights(matrix)
        assert w.shape == (4,)
        assert np.all(w >= 0)
        np.testing.assert_allclose(w.sum(), 1.0)


class TestPCAWeights:
    def test_basic(self):
        rng = np.random.default_rng(42)
        matrix = rng.random((50, 3))
        w = pca_weights(matrix)
        assert w.shape == (3,)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_too_few_samples(self):
        matrix = np.random.random((2, 5))
        w = pca_weights(matrix)
        np.testing.assert_allclose(w, np.ones(5) / 5)


class TestMCDAScores:
    def test_basic(self):
        grid_points = [
            {"cf": 0.25, "slope": 10.0},
            {"cf": 0.20, "slope": 15.0},
            {"cf": 0.30, "slope": 2.0},
        ]
        criteria = {
            "cf": {"enabled": True, "weight": 0.7, "direction": "maximize"},
            "slope": {"enabled": True, "weight": 0.3, "direction": "minimize"},
        }
        scores, weights = compute_mcda_scores(grid_points, criteria, method="manual")

        assert len(scores) == 3
        assert len(weights) == 2
        # Best should be point 2 (highest cf, lowest slope)
        assert np.argmax(scores) == 2

    def test_empty(self):
        scores, weights = compute_mcda_scores([], {}, method="manual")
        assert len(scores) == 0
        assert len(weights) == 0
