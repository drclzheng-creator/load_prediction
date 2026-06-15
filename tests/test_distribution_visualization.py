from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DISTRIBUTION_CSV = Path(
    "outputs/gmm/day_ahead/e2e_probability_plots/probability/point_scenarios.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/distribution_visualization/artifacts/distribution_samples.png"
)
SPECIFIC_ROWS_OUTPUT_PATH = Path(
    "outputs/distribution_visualization/artifacts/distribution_selected_rows.png"
)
POINT_SCENARIO_OUTPUT_PATH = Path(
    "outputs/distribution_visualization/artifacts/point_scenario_distributions.png"
)
POINT_SCENARIO_TIMELINE_OUTPUT_PATH = Path(
    "outputs/distribution_visualization/artifacts/point_scenario_timeline.png"
)


def _load_distribution_frame() -> pd.DataFrame:
    csv_path = Path(
        os.environ.get("LOAD_PREDICTION_DISTRIBUTION_CSV", DEFAULT_DISTRIBUTION_CSV)
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Distribution forecast CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"distribution_values", "distribution_probabilities"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Distribution CSV missing columns: {sorted(missing)}")
    return frame


def _parse_distribution(row: pd.Series) -> tuple[list[float], list[float]]:
    values = [float(value) for value in json.loads(row["distribution_values"])]
    probabilities = [
        float(value) for value in json.loads(row["distribution_probabilities"])
    ]
    if len(values) != len(probabilities):
        raise AssertionError(
            "distribution_values and distribution_probabilities must have equal length"
        )
    if len(values) < 2:
        raise AssertionError("Distribution grid must contain at least two points")
    if any(probability < 0 for probability in probabilities):
        raise AssertionError("Distribution probabilities must be non-negative")
    return values, probabilities


def _parse_json_array(row: pd.Series, column: str) -> list[float]:
    if column not in row or pd.isna(row[column]):
        return []
    values = json.loads(row[column])
    return [float(value) for value in values]


def _detect_mixture_prefix(frame: pd.DataFrame) -> str | None:
    for prefix in ("gmm", "mdn", "lstm"):
        if {f"{prefix}_weights", f"{prefix}_means", f"{prefix}_stds"}.issubset(frame.columns):
            return prefix
    return None


def _normal_pdf(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    z = (values - float(mean)) / std
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2 * np.pi))


def _selected_row_indices() -> list[int] | None:
    raw_rows = os.environ.get("LOAD_PREDICTION_DISTRIBUTION_ROWS", "").strip()
    if not raw_rows:
        return None
    try:
        return [int(value.strip()) for value in raw_rows.split(",") if value.strip()]
    except ValueError as exc:
        raise AssertionError(
            "LOAD_PREDICTION_DISTRIBUTION_ROWS must be comma-separated integer row indices"
        ) from exc


def _select_distribution_rows(
    frame: pd.DataFrame,
    sample_size: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Path]:
    selected_indices = _selected_row_indices()
    if selected_indices is None:
        sample_count = min(sample_size, len(frame))
        if sample_count <= 0:
            raise AssertionError("Distribution frame is empty")
        return (
            frame.sample(n=sample_count, random_state=random_state).copy(),
            DEFAULT_OUTPUT_PATH,
        )

    if not selected_indices:
        raise AssertionError("No valid row indices were provided")
    invalid = [index for index in selected_indices if index < 0 or index >= len(frame)]
    if invalid:
        raise AssertionError(
            f"Selected row indices out of range: {invalid}; valid range is 0..{len(frame) - 1}"
        )
    return frame.iloc[selected_indices].copy(), SPECIFIC_ROWS_OUTPUT_PATH


def _plot_distribution_samples(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Distribution visualization requires matplotlib") from exc

    sample_count = len(frame)
    if sample_count <= 0:
        raise AssertionError("Distribution frame is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(sample_count, 1, figsize=(8, 2.4 * sample_count), sharex=False)
    axes = [axes] if sample_count == 1 else list(axes)
    for plot_index, (axis, (row_index, row)) in enumerate(zip(axes, frame.iterrows()), 1):
        values, probabilities = _parse_distribution(row)
        axis.plot(values, probabilities, color="#1f77b4", linewidth=1.8)
        axis.fill_between(values, probabilities, color="#1f77b4", alpha=0.16)
        if "prediction" in row and pd.notna(row["prediction"]):
            axis.axvline(
                float(row["prediction"]),
                color="#d62728",
                linewidth=1.3,
                label="Prediction",
            )
            axis.legend(loc="best")
        title = f"Probability Distribution Row {row_index}"
        if "timestamp" in row and pd.notna(row["timestamp"]):
            title = f"{title} - {row['timestamp']}"
        axis.set_title(title)
        axis.set_xlabel("Load")
        axis.set_ylabel("Density")
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_point_scenario_distribution_samples(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Distribution visualization requires matplotlib") from exc

    sample_count = len(frame)
    if sample_count <= 0:
        raise AssertionError("Point scenario frame is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixture_prefix = _detect_mixture_prefix(frame)
    component_colors = plt.cm.Set2(np.linspace(0, 1, 8))

    fig, axes = plt.subplots(sample_count, 1, figsize=(10, 3.2 * sample_count), sharex=False)
    axes = [axes] if sample_count == 1 else list(axes)
    for axis, (row_index, row) in zip(axes, frame.iterrows()):
        values, probabilities = _parse_distribution(row)
        values_array = np.asarray(values, dtype=float)
        probabilities_array = np.asarray(probabilities, dtype=float)
        max_density = float(np.nanmax(probabilities_array)) if len(probabilities_array) else 1.0
        axis.plot(values_array, probabilities_array, color="#1f77b4", linewidth=1.8, label="Mixture PDF")
        axis.fill_between(values_array, probabilities_array, color="#1f77b4", alpha=0.14)

        if mixture_prefix is not None:
            weights = _parse_json_array(row, f"{mixture_prefix}_weights")
            means = _parse_json_array(row, f"{mixture_prefix}_means")
            stds = _parse_json_array(row, f"{mixture_prefix}_stds")
            for component_index, (weight, mean, std) in enumerate(zip(weights, means, stds)):
                color = component_colors[component_index % len(component_colors)]
                component_pdf = float(weight) * _normal_pdf(values_array, mean, std)
                axis.plot(
                    values_array,
                    component_pdf,
                    color=color,
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"{mixture_prefix.upper()} component {component_index + 1}",
                )
                axis.axvline(float(mean), color=color, linewidth=0.9, alpha=0.45, linestyle=":")

        scenario_values = _parse_json_array(row, "point_scenario_values")
        scenario_probabilities = _parse_json_array(row, "point_scenario_probabilities")
        if scenario_values and scenario_probabilities:
            scenario_values_array = np.asarray(scenario_values, dtype=float)
            scenario_probabilities_array = np.asarray(scenario_probabilities, dtype=float)
            scale = max_density / max(float(np.nanmax(scenario_probabilities_array)), 1e-12)
            markerline, stemlines, baseline = axis.stem(
                scenario_values_array,
                scenario_probabilities_array * scale,
                linefmt="k-",
                markerfmt="ko",
                basefmt=" ",
                label="Reduced scenarios",
            )
            plt.setp(stemlines, linewidth=1.1, alpha=0.75)
            plt.setp(markerline, markersize=4.5, alpha=0.9)
            for value, probability in zip(scenario_values_array, scenario_probabilities_array):
                axis.annotate(
                    f"{probability:.2f}",
                    xy=(float(value), float(probability * scale)),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#111111",
                )

        if "prediction" in row and pd.notna(row["prediction"]):
            axis.axvline(float(row["prediction"]), color="#d62728", linewidth=1.3, label="Prediction")
        if "0.1" in row and pd.notna(row["0.1"]):
            axis.axvline(float(row["0.1"]), color="#666666", linewidth=1.0, linestyle="--", label="P10/P90")
        if "0.9" in row and pd.notna(row["0.9"]):
            axis.axvline(float(row["0.9"]), color="#666666", linewidth=1.0, linestyle="--")

        title = f"Point Scenario Distribution Row {row_index}"
        if "timestamp" in row and pd.notna(row["timestamp"]):
            title = f"{title} - {row['timestamp']}"
        axis.set_title(title)
        axis.set_xlabel("Load")
        axis.set_ylabel("Density / scaled scenario mass")
        axis.grid(True, alpha=0.24)
        handles, labels = axis.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        axis.legend(deduped.values(), deduped.keys(), loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_point_scenario_timeline(
    frame: pd.DataFrame,
    output_path: Path,
    max_rows: int = 96,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Distribution visualization requires matplotlib") from exc

    if frame.empty:
        raise AssertionError("Point scenario frame is empty")
    required = {"timestamp", "prediction", "point_scenario_values", "point_scenario_probabilities"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Point scenario timeline missing columns: {sorted(missing)}")

    plot_frame = frame.head(max_rows).copy()
    plot_frame["timestamp"] = pd.to_datetime(plot_frame["timestamp"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(1, 1, figsize=(12, 4.8))
    axis.plot(plot_frame["timestamp"], plot_frame["prediction"], color="#111111", linewidth=1.6, label="Prediction")
    if {"0.1", "0.9"}.issubset(plot_frame.columns):
        axis.fill_between(
            plot_frame["timestamp"],
            plot_frame["0.1"].astype(float),
            plot_frame["0.9"].astype(float),
            color="#1f77b4",
            alpha=0.16,
            label="P10-P90",
        )

    timestamps = []
    scenario_values = []
    scenario_probabilities = []
    for _, row in plot_frame.iterrows():
        values = _parse_json_array(row, "point_scenario_values")
        probabilities = _parse_json_array(row, "point_scenario_probabilities")
        timestamps.extend([row["timestamp"]] * len(values))
        scenario_values.extend(values)
        scenario_probabilities.extend(probabilities)
    probabilities_array = np.asarray(scenario_probabilities, dtype=float)
    sizes = 20 + 260 * probabilities_array / max(float(probabilities_array.max()), 1e-12)
    scatter = axis.scatter(
        timestamps,
        scenario_values,
        s=sizes,
        c=probabilities_array,
        cmap="viridis",
        alpha=0.72,
        edgecolors="none",
        label="Point scenarios",
    )
    fig.colorbar(scatter, ax=axis, label="Scenario probability")
    axis.set_title("Point Scenario Timeline")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Load")
    axis.grid(True, alpha=0.24)
    axis.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def test_distribution_values_probabilities_can_be_visualized():
    frame = _load_distribution_frame()
    selected_frame, output_path = _select_distribution_rows(frame)
    output_path = _plot_distribution_samples(selected_frame, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_point_scenario_distribution_can_be_visualized():
    frame = _load_distribution_frame()
    required = {"point_scenario_values", "point_scenario_probabilities"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Point scenario CSV missing columns: {sorted(missing)}")
    selected_frame, _ = _select_distribution_rows(frame)
    output_path = _plot_point_scenario_distribution_samples(
        selected_frame,
        POINT_SCENARIO_OUTPUT_PATH,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_point_scenario_timeline_can_be_visualized():
    frame = _load_distribution_frame()
    output_path = _plot_point_scenario_timeline(
        frame,
        POINT_SCENARIO_TIMELINE_OUTPUT_PATH,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
