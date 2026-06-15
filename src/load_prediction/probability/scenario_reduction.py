"""Sampling and reduction primitives for point scenarios."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from load_prediction.configs import PointScenarioConfig


def point_scenarios_from_mixture(
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    config: PointScenarioConfig,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights, means, stds = validate_mixture(weights, means, stds)
    rng = np.random.default_rng(random_state)
    component_indices = rng.choice(len(weights), size=config.num_samples, p=weights)
    samples = rng.normal(means[component_indices], stds[component_indices])
    return reduce_samples_to_scenarios(samples, config, random_state, rng)


def point_scenarios_from_quantiles(
    levels: np.ndarray,
    values: np.ndarray,
    config: PointScenarioConfig,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    levels, values = validate_quantile_function(levels, values)
    rng = np.random.default_rng(random_state)
    sample_levels = rng.random(config.num_samples)
    samples = np.interp(sample_levels, levels, values)
    return reduce_samples_to_scenarios(samples, config, random_state, rng)


def point_scenarios_from_density_grid(
    values: np.ndarray,
    densities: np.ndarray,
    config: PointScenarioConfig,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    values, masses = density_grid_to_mass(values, densities)
    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=config.num_samples, p=masses)
    return reduce_samples_to_scenarios(samples, config, random_state, rng)


def reduce_samples_to_scenarios(
    samples: np.ndarray,
    config: PointScenarioConfig,
    random_state: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if config.clip_min is not None:
        samples = np.clip(samples, float(config.clip_min), None)

    fit_samples = samples
    if len(samples) > config.kmeans_fit_samples:
        fit_indices = rng.choice(len(samples), size=config.kmeans_fit_samples, replace=False)
        fit_samples = samples[fit_indices]

    kmeans = KMeans(
        n_clusters=config.num_scenarios,
        random_state=random_state,
        init="random",
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
    )
    kmeans.fit(fit_samples.reshape(-1, 1))
    centers = kmeans.cluster_centers_.reshape(-1)
    labels = np.argmin(np.abs(samples.reshape(-1, 1) - centers.reshape(1, -1)), axis=1)
    probabilities = np.bincount(labels, minlength=config.num_scenarios).astype(float)
    probabilities = probabilities / probabilities.sum()

    order = np.argsort(centers)
    return centers[order].astype(float), probabilities[order].astype(float)


def density_grid_to_mass(values: np.ndarray, densities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != len(densities):
        raise ValueError(
            "distribution_values and distribution_probabilities must have the same length; "
            f"got {len(values)} and {len(densities)}"
        )
    if len(values) < 2:
        raise ValueError("Distribution grid must contain at least two points")
    order = np.argsort(values)
    values = values[order].astype(float)
    densities = np.clip(densities[order].astype(float), 0.0, None)
    if not np.isfinite(values).all() or not np.isfinite(densities).all():
        raise ValueError("Distribution grid contains non-finite values")
    widths = np.empty_like(values, dtype=float)
    widths[0] = max((values[1] - values[0]) / 2.0, 1e-12)
    widths[-1] = max((values[-1] - values[-2]) / 2.0, 1e-12)
    if len(values) > 2:
        widths[1:-1] = np.maximum((values[2:] - values[:-2]) / 2.0, 1e-12)
    masses = densities * widths
    total_mass = float(masses.sum())
    if total_mass <= 0:
        masses = np.full(len(values), 1.0 / len(values), dtype=float)
    else:
        masses = masses / total_mass
    return values, masses


def validate_point_scenario_config(config: PointScenarioConfig) -> None:
    if config.num_scenarios < 1:
        raise ValueError("num_scenarios must be at least 1")
    if config.num_samples < config.num_scenarios:
        raise ValueError("num_samples must be greater than or equal to num_scenarios")
    if config.kmeans_n_init < 1:
        raise ValueError("kmeans_n_init must be at least 1")
    if config.kmeans_max_iter < 1:
        raise ValueError("kmeans_max_iter must be at least 1")
    if config.kmeans_fit_samples < config.num_scenarios:
        raise ValueError("kmeans_fit_samples must be greater than or equal to num_scenarios")
    if not config.values_col:
        raise ValueError("values_col must be non-empty")
    if not config.probabilities_col:
        raise ValueError("probabilities_col must be non-empty")


def validate_quantile_function(
    levels: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(levels, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(levels) != len(values):
        raise ValueError(
            "Quantile levels and values must have the same length; "
            f"got {len(levels)} and {len(values)}"
        )
    if len(levels) < 2:
        raise ValueError("Quantile function requires at least two levels")
    if not np.isfinite(levels).all() or not np.isfinite(values).all():
        raise ValueError("Quantile function contains non-finite values")
    if ((levels <= 0.0) | (levels >= 1.0)).any():
        raise ValueError("Quantile levels must be between 0 and 1")
    order = np.argsort(levels)
    levels = levels[order]
    values = values[order]
    unique_levels, unique_indices = np.unique(levels, return_index=True)
    values = values[unique_indices]
    values = np.maximum.accumulate(values)
    if unique_levels[0] > 0.0:
        unique_levels = np.concatenate([[0.0], unique_levels])
        values = np.concatenate([[values[0]], values])
    if unique_levels[-1] < 1.0:
        unique_levels = np.concatenate([unique_levels, [1.0]])
        values = np.concatenate([values, [values[-1]]])
    return unique_levels.astype(float), values.astype(float)


def validate_mixture(
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (len(weights) == len(means) == len(stds)):
        raise ValueError(
            "Mixture weights, means, and stds must have the same length; "
            f"got {len(weights)}, {len(means)}, {len(stds)}"
        )
    if (weights < 0).any():
        raise ValueError("Mixture weights must be non-negative")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Mixture weights must sum to a positive value")
    if (stds <= 0).any():
        raise ValueError("Mixture stds must be positive")
    return (
        (weights / total_weight).astype(float),
        means.astype(float),
        stds.astype(float),
    )

