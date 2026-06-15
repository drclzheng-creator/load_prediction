"""Sklearn tabular estimator adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

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
class RecursiveSklearnForecaster(RecursiveTabularForecaster):
    """Sklearn estimators behind the shared recursive tabular forecaster."""

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        feature_config: FeatureEngineeringConfig,
        estimator_name: str = "random_forest",
    ) -> "RecursiveSklearnForecaster":
        params = dict(model_config.params)
        logger.info(
            "Creating sklearn estimator=%s random_state=%s scale_features=%s scale_target=%s",
            estimator_name,
            model_config.random_state,
            model_config.scale_features,
            model_config.scale_target,
        )
        if estimator_name == "hist_gradient_boosting":
            enable_quantiles = params.pop("enable_quantiles", True)
            lower_quantile = params.pop("lower_quantile", DEFAULT_INTERVAL_LOWER_QUANTILE)
            upper_quantile = params.pop("upper_quantile", DEFAULT_INTERVAL_UPPER_QUANTILE)
            quantile_levels = tuple(
                float(level)
                for level in params.pop("quantile_levels", DEFAULT_QUANTILE_LEVELS)
            )
            quantile_levels = tuple(
                sorted({*quantile_levels, float(lower_quantile), float(upper_quantile)})
            )
            if enable_quantiles:
                point_loss = params.pop("loss", "absolute_error")
                logger.info(
                    "Using HistGradientBoostingRegressor point loss=%s and quantiles=%s",
                    point_loss,
                    quantile_levels,
                )
                point_estimator = HistGradientBoostingRegressor(
                    loss=point_loss,
                    random_state=model_config.random_state,
                    **params,
                )

                def quantile_estimator_factory(quantile: float):
                    return HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=quantile,
                        random_state=model_config.random_state,
                        **params,
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
            loss = params.pop("loss", "absolute_error")
            logger.info("Using HistGradientBoostingRegressor loss=%s", loss)
            estimator = HistGradientBoostingRegressor(
                loss=loss,
                random_state=model_config.random_state,
                **params,
            )
        else:
            criterion = params.pop("criterion", "absolute_error")
            logger.info("Using RandomForestRegressor criterion=%s", criterion)
            estimator = RandomForestRegressor(
                n_estimators=params.pop("n_estimators", 160),
                criterion=criterion,
                min_samples_leaf=params.pop("min_samples_leaf", 2),
                n_jobs=params.pop("n_jobs", -1),
                random_state=model_config.random_state,
                **params,
            )
        return cls(
            feature_builder=FeatureBuilder(feature_config),
            estimator=estimator,
            scale_features=model_config.scale_features,
            scale_target=model_config.scale_target,
        )
