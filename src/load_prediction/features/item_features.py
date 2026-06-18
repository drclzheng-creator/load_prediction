"""Item identity feature builders."""

from __future__ import annotations

import pandas as pd


def build_item_feature_frame(
    item_id: str,
    item_ids: list[str],
    *,
    length: int,
    enabled: bool = True,
) -> pd.DataFrame:
    features = pd.DataFrame(index=range(length))
    if not enabled:
        return features
    for known_item in item_ids:
        features[f"feature__item__{known_item}"] = float(str(item_id) == known_item)
    return features


def build_item_features(
    item_id: str,
    item_ids: list[str],
    *,
    enabled: bool = True,
) -> dict[str, float]:
    if not enabled:
        return {}
    return {f"feature__item__{known_item}": float(str(item_id) == known_item) for known_item in item_ids}
