"""Post-processing for forecast outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from statistics import NormalDist

import numpy as np

from load_prediction.configs import ForecastPostprocessingConfig
from load_prediction.models.base_forecaster import ForecastFrame

logger = logging.getLogger(__name__)

NORMAL = NormalDist()
P10_Z = NORMAL.inv_cdf(0.1)
P90_Z = NORMAL.inv_cdf(0.9)


@dataclass
class ForecastPostProcessor:
    config: ForecastPostprocessingConfig

    def transform(self, forecast: ForecastFrame, residual_std: float = 0.0) -> ForecastFrame:
        frame = forecast.frame.copy()
        pred_col = forecast.prediction_col
        if self.config.clip_negative:
            negative_count = int((frame[pred_col] < 0).sum())
            if negative_count:
                logger.warning("Clipping %s negative forecast values to zero", negative_count)
            frame[pred_col] = frame[pred_col].clip(lower=0)

        if self.config.add_residual_interval:
            if {"prediction_lower", "prediction_upper"}.issubset(frame.columns):
                logger.info("Using model-provided prediction interval")
            else:
                margin = self.config.interval_width * residual_std
                frame["prediction_lower"] = (frame[pred_col] - margin).clip(lower=0)
                frame["prediction_upper"] = frame[pred_col] + margin
                logger.info("Added residual prediction interval width=%s residual_std=%.6f", self.config.interval_width, residual_std)

        if self.config.add_distribution_grid:
            frame = self._add_distribution_grid(frame, pred_col)

        return ForecastFrame(
            frame=frame,
            timestamp_col=forecast.timestamp_col,
            item_id_col=forecast.item_id_col,
            prediction_col=forecast.prediction_col,
        )

    def _add_distribution_grid(self, frame, pred_col: str):
        if {"distribution_values", "distribution_probabilities"}.issubset(frame.columns):
            logger.info("Using model-provided probability distribution grid")
            return frame
        if pred_col not in frame.columns:
            logger.warning("Skipping distribution grid: missing prediction column=%s", pred_col)
            return frame

        quantile_columns = _detect_quantile_columns(frame)
        if len(quantile_columns) >= 3:
            logger.info(
                "Adding distribution grid from quantile function columns=%s",
                [column for column, _ in quantile_columns],
            )
            return self._add_quantile_distribution_grid(frame, pred_col, quantile_columns)

        if {"0.1", "0.9"}.issubset(frame.columns):
            lower_col = "0.1"
            upper_col = "0.9"
            z_width = P90_Z - P10_Z
            lower_z = P10_Z
            logger.info("Adding normal distribution grid from P10/P90 quantile columns")
        elif {"prediction_lower", "prediction_upper"}.issubset(frame.columns):
            lower_col = "prediction_lower"
            upper_col = "prediction_upper"
            z_width = 2 * self.config.interval_width
            lower_z = -self.config.interval_width
            logger.info("Adding normal distribution grid from prediction interval columns")
        else:
            logger.info("Skipping distribution grid: no interval columns available")
            return frame

        grid_size = max(int(self.config.distribution_grid_size), 2)
        std_width = float(self.config.distribution_std_width)
        distribution_values = []
        distribution_probabilities = []
        prediction_density = []
        for row in frame[[pred_col, lower_col, upper_col]].itertuples(index=False):
            prediction = float(row[0])
            lower = float(row[1])
            upper = float(row[2])
            sigma = max((upper - lower) / z_width, 1e-6)
            mean = lower - lower_z * sigma
            grid_low = max(0.0, mean - std_width * sigma)
            grid_high = mean + std_width * sigma
            grid_low = min(grid_low, prediction)
            grid_high = max(grid_high, prediction)
            if grid_high <= grid_low:
                grid_high = grid_low + 1e-6
            values = [
                grid_low + (grid_high - grid_low) * index / (grid_size - 1)
                for index in range(grid_size)
            ]
            probabilities = [NORMAL.pdf((value - mean) / sigma) / sigma for value in values]
            distribution_values.append(json.dumps(values))
            distribution_probabilities.append(json.dumps(probabilities))
            prediction_density.append(NORMAL.pdf((prediction - mean) / sigma) / sigma)

        frame = frame.copy()
        frame["distribution_values"] = distribution_values
        frame["distribution_probabilities"] = distribution_probabilities
        if "prediction_density" not in frame.columns:
            frame["prediction_density"] = prediction_density
        return frame

    def _add_quantile_distribution_grid(
        self,
        frame,
        pred_col: str,
        quantile_columns: list[tuple[str, float]],
    ):
        grid_size = max(int(self.config.distribution_grid_size), 2)
        tail_eps = min(1e-4, min(level for _, level in quantile_columns) / 2)
        base_levels = np.asarray([level for _, level in quantile_columns], dtype=float)
        columns = [column for column, _ in quantile_columns]
        distribution_values = []
        distribution_probabilities = []
        prediction_density = []

        for row in frame[[pred_col, *columns]].itertuples(index=False):
            prediction = float(row[0])
            quantile_values = np.asarray([float(value) for value in row[1:]], dtype=float)
            quantile_values = np.maximum.accumulate(quantile_values)
            unique_levels, unique_indices = np.unique(base_levels, return_index=True)
            unique_values = quantile_values[unique_indices]
            unique_values = np.maximum.accumulate(unique_values)
            if np.unique(unique_values).size < 2:
                values, probabilities, pred_density = self._normal_grid_from_point(prediction)
            else:
                left_slope = _safe_quantile_slope(
                    unique_values[0],
                    unique_values[1],
                    unique_levels[0],
                    unique_levels[1],
                )
                right_slope = _safe_quantile_slope(
                    unique_values[-2],
                    unique_values[-1],
                    unique_levels[-2],
                    unique_levels[-1],
                )
                low_level = tail_eps
                high_level = 1.0 - tail_eps
                low_value = max(
                    0.0,
                    unique_values[0] - left_slope * max(unique_levels[0] - low_level, 0.0),
                )
                high_value = unique_values[-1] + right_slope * max(high_level - unique_levels[-1], 0.0)
                grid_levels = np.linspace(low_level, high_level, grid_size)
                q_levels = np.concatenate([[low_level], unique_levels, [high_level]])
                q_values = np.concatenate([[low_value], unique_values, [high_value]])
                values = np.interp(grid_levels, q_levels, q_values)
                values = np.maximum.accumulate(np.clip(values, 0.0, None))
                probabilities = _density_from_quantile_grid(values, grid_levels)
                pred_density = float(np.interp(prediction, values, probabilities))

            distribution_values.append(json.dumps([float(value) for value in values]))
            distribution_probabilities.append(json.dumps([float(value) for value in probabilities]))
            prediction_density.append(float(pred_density))

        frame = frame.copy()
        frame["distribution_values"] = distribution_values
        frame["distribution_probabilities"] = distribution_probabilities
        if "prediction_density" not in frame.columns:
            frame["prediction_density"] = prediction_density
        return frame

    def _normal_grid_from_point(self, prediction: float):
        grid_size = max(int(self.config.distribution_grid_size), 2)
        sigma = max(abs(prediction) * 0.01, 1e-6)
        std_width = float(self.config.distribution_std_width)
        grid_low = max(0.0, prediction - std_width * sigma)
        grid_high = max(prediction + std_width * sigma, grid_low + 1e-6)
        values = np.linspace(grid_low, grid_high, grid_size)
        probabilities = np.asarray(
            [NORMAL.pdf((value - prediction) / sigma) / sigma for value in values],
            dtype=float,
        )
        pred_density = NORMAL.pdf(0.0) / sigma
        return values, probabilities, pred_density


def _detect_quantile_columns(frame) -> list[tuple[str, float]]:
    quantile_columns: list[tuple[str, float]] = []
    for column in frame.columns:
        try:
            level = float(column)
        except (TypeError, ValueError):
            continue
        if 0.0 < level < 1.0:
            quantile_columns.append((column, level))
    return sorted(quantile_columns, key=lambda item: item[1])


def _safe_quantile_slope(
    value_a: float,
    value_b: float,
    level_a: float,
    level_b: float,
) -> float:
    level_gap = max(level_b - level_a, 1e-6)
    return max((value_b - value_a) / level_gap, 1e-6)


def _density_from_quantile_grid(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    density = np.zeros_like(values, dtype=float)
    for index in range(len(values)):
        if index == 0:
            value_width = values[1] - values[0]
            level_width = levels[1] - levels[0]
        elif index == len(values) - 1:
            value_width = values[-1] - values[-2]
            level_width = levels[-1] - levels[-2]
        else:
            value_width = values[index + 1] - values[index - 1]
            level_width = levels[index + 1] - levels[index - 1]
        density[index] = level_width / max(value_width, 1e-6)
    return density
