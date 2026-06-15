"""Train/test split helpers for forecasting pipelines."""

from __future__ import annotations

import logging

import pandas as pd

from load_prediction.configs import PipelineConfig
from load_prediction.data.data_schema import TimeSeriesDataset

logger = logging.getLogger(__name__)


def split_train_test(
    data: TimeSeriesDataset,
    config: PipelineConfig,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
    evaluation = config.evaluation
    if evaluation.train_ratio is not None:
        logger.info(
            "Splitting by train_ratio=%.4f aligned_to_prediction_length=%s",
            evaluation.train_ratio,
            config.scale.prediction_length,
        )
        return split_by_aligned_ratio(
            data,
            evaluation.train_ratio,
            config.scale.prediction_length,
        )
    if evaluation.holdout_length is not None:
        logger.info("Splitting by fixed holdout_length=%s", evaluation.holdout_length)
        return data.split_holdout(evaluation.holdout_length)
    if evaluation.split_strategy == "prediction_length":
        logger.info("Splitting by scale prediction_length=%s", config.scale.prediction_length)
        return data.split_holdout(config.scale.prediction_length)
    raise ValueError(
        "Unsupported split_strategy "
        f"'{evaluation.split_strategy}'. Use ratio or prediction_length."
    )


def split_by_aligned_ratio(
    data: TimeSeriesDataset,
    train_ratio: float,
    align_length: int,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if align_length <= 0:
        raise ValueError("align_length must be positive")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for item_id, group in data.frame.groupby(data.item_id_col, sort=False):
        raw_split = int(len(group) * train_ratio)
        aligned_split = int(round(raw_split / align_length) * align_length)
        if aligned_split <= 0 or aligned_split >= len(group):
            raise ValueError(
                f"Not enough rows for train_ratio={train_ratio}; "
                f"item_id={item_id} has only {len(group)} rows."
            )
        if aligned_split != raw_split:
            logger.info(
                "Aligned train/test split for item_id=%s raw_split=%s aligned_split=%s align_length=%s",
                item_id,
                raw_split,
                aligned_split,
                align_length,
            )
        train_parts.append(group.iloc[:aligned_split])
        test_parts.append(group.iloc[aligned_split:])

    return data.with_frame(pd.concat(train_parts, ignore_index=True)), data.with_frame(
        pd.concat(test_parts, ignore_index=True)
    )

