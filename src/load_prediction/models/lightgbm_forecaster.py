"""LightGBM model adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

from load_prediction.configs import FeatureEngineeringConfig, ModelSpecConfig
from load_prediction.constants import (
    DEFAULT_INTERVAL_LOWER_QUANTILE,
    DEFAULT_INTERVAL_UPPER_QUANTILE,
    DEFAULT_QUANTILE_LEVELS,
)
from load_prediction.features import FeatureBuilder
from load_prediction.models.recursive_tabular import (
    QuantileRecursiveTabularForecaster,
    RecursiveTabularForecaster,
)

logger = logging.getLogger(__name__)

@dataclass
class LightGBMForecaster(RecursiveTabularForecaster):
    """LightGBM implementation behind the BaseForecastModel contract.

    It reuses the recursive forecasting mechanics from the sklearn adapter while
    swapping only the underlying one-step estimator.
    """

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        feature_config: FeatureEngineeringConfig,
    ) -> "LightGBMForecaster":
        mpl_config_dir = Path("tmp/matplotlib")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBM model requires: pip install lightgbm") from exc

        params = dict(model_config.params)
        enable_quantiles = params.pop("enable_quantiles", True)
        lower_quantile = params.pop("lower_quantile", DEFAULT_INTERVAL_LOWER_QUANTILE)
        upper_quantile = params.pop("upper_quantile", DEFAULT_INTERVAL_UPPER_QUANTILE)
        quantile_levels = tuple(
            float(level)
            for level in params.pop("quantile_levels", DEFAULT_QUANTILE_LEVELS)
        )
        quantile_levels = tuple(sorted({*quantile_levels, float(lower_quantile), float(upper_quantile)}))
        objective = params.pop("objective", "regression_l1")
        base_params = {
            "n_estimators": params.pop("n_estimators", 300),
            "learning_rate": params.pop("learning_rate", 0.05),
            "num_leaves": params.pop("num_leaves", 31),
            "n_jobs": params.pop("n_jobs", 1),
            "verbosity": params.pop("verbosity", -1),
            **params,
        }
        if enable_quantiles:
            logger.info(
                "Creating LightGBM point estimator objective=%s and quantile estimators=%s random_state=%s",
                objective,
                quantile_levels,
                model_config.random_state,
            )

            point_estimator = LGBMRegressor(
                **base_params,
                objective=objective,
                random_state=model_config.random_state,
            )

            def quantile_estimator_factory(alpha: float):
                return LGBMRegressor(
                    **base_params,
                    objective="quantile",
                    alpha=alpha,
                    random_state=model_config.random_state,
                )

            return QuantileRecursiveTabularForecaster(
                feature_builder=FeatureBuilder(feature_config),
                estimator=point_estimator,
                lower_estimator=quantile_estimator_factory(lower_quantile),
                upper_estimator=quantile_estimator_factory(upper_quantile),
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                quantile_levels=quantile_levels,
                quantile_estimator_factory=quantile_estimator_factory,
                scale_features=model_config.scale_features,
                scale_target=model_config.scale_target,
            )
        logger.info(
            "Creating LightGBM estimator objective=%s random_state=%s scale_features=%s scale_target=%s",
            objective,
            model_config.random_state,
            model_config.scale_features,
            model_config.scale_target,
        )
        estimator = LGBMRegressor(
            n_estimators=base_params.pop("n_estimators"),
            learning_rate=base_params.pop("learning_rate"),
            num_leaves=base_params.pop("num_leaves"),
            objective=objective,
            random_state=model_config.random_state,
            n_jobs=base_params.pop("n_jobs"),
            verbosity=base_params.pop("verbosity"),
            **base_params,
        )
        return cls(
            feature_builder=FeatureBuilder(feature_config),
            estimator=estimator,
            scale_features=model_config.scale_features,
            scale_target=model_config.scale_target,
        )
