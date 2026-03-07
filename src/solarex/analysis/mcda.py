"""Multi-criteria decision analysis engine for solar PV site selection.

Supports manual, entropy (Shannon), and PCA weighting methods.
No sklearn dependency — PCA uses SVD via numpy.
"""

from __future__ import annotations

import numpy as np


def entropy_weights(norm_matrix: np.ndarray) -> np.ndarray:
    """Compute weights using Shannon entropy method.

    Parameters
    ----------
    norm_matrix : np.ndarray
        Normalized criteria matrix (n_sites, n_criteria) with values in [0, 1].

    Returns
    -------
    np.ndarray
        Weights summing to 1.0.
    """
    n, m = norm_matrix.shape
    if n <= 1:
        return np.ones(m) / m

    shifted = norm_matrix + 1e-10
    col_sums = shifted.sum(axis=0)
    col_sums[col_sums == 0] = 1
    p = shifted / col_sums

    k = 1.0 / np.log(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -k * np.nansum(p * np.log(p + 1e-30), axis=0)

    d = 1.0 - H
    d = np.maximum(d, 0)

    total = d.sum()
    if total > 0:
        return d / total
    return np.ones(m) / m


def pca_weights(norm_matrix: np.ndarray) -> np.ndarray:
    """Compute weights from first principal component loadings.

    Uses SVD-based PCA (no sklearn dependency).

    Parameters
    ----------
    norm_matrix : np.ndarray
        Normalized criteria matrix (n_sites, n_criteria).

    Returns
    -------
    np.ndarray
        Weights summing to 1.0.
    """
    n, m = norm_matrix.shape
    if n <= m or m <= 1:
        return np.ones(m) / m

    std = norm_matrix.std(axis=0)
    std[std == 0] = 1
    standardized = (norm_matrix - norm_matrix.mean(axis=0)) / std

    # SVD-based PCA
    _, _, Vt = np.linalg.svd(standardized, full_matrices=False)
    loadings = np.abs(Vt[0])

    total = loadings.sum()
    if total > 0:
        return loadings / total
    return np.ones(m) / m


def compute_mcda_scores(
    grid_points: list[dict],
    criteria: dict,
    method: str = "manual",
) -> tuple[np.ndarray, np.ndarray]:
    """Run MCDA scoring on grid points.

    Parameters
    ----------
    grid_points : list[dict]
        Each dict has criterion values (e.g. "capacity_factor", "slope").
    criteria : dict
        ``{name: {"enabled": bool, "weight": float, "direction": "maximize"|"minimize"}}``.
    method : str
        "manual", "entropy", or "pca".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, weights) arrays.
    """
    enabled = {name: c for name, c in criteria.items() if c.get("enabled", True)}
    criteria_names = list(enabled.keys())
    n_cells = len(grid_points)
    n_criteria = len(criteria_names)

    if n_cells == 0 or n_criteria == 0:
        return np.array([]), np.array([])

    # Build raw matrix
    raw_matrix = np.zeros((n_cells, n_criteria))
    for j, name in enumerate(criteria_names):
        for i, pt in enumerate(grid_points):
            raw_matrix[i, j] = pt.get(name, 0.0)

    # Min-max normalize
    norm_matrix = np.zeros_like(raw_matrix)
    for j in range(n_criteria):
        col = raw_matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-10:
            norm_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            norm_matrix[:, j] = 0.5

        # Invert for "minimize" criteria
        if enabled[criteria_names[j]].get("direction", "maximize") == "minimize":
            norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

    # Compute weights
    if method == "entropy":
        weights = entropy_weights(norm_matrix)
    elif method == "pca":
        weights = pca_weights(norm_matrix)
    else:
        raw_w = np.array([
            enabled[name].get("weight", 1.0 / n_criteria) for name in criteria_names
        ])
        total_w = raw_w.sum()
        weights = raw_w / total_w if total_w > 0 else np.ones(n_criteria) / n_criteria

    scores = norm_matrix @ weights
    return scores, weights
