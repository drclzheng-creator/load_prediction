"""Pipeline orchestration for load forecasting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from load_prediction.configs import PipelineConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.evaluation import evaluate_forecast
from load_prediction.inference.model_manifest import build_model_manifest, save_model_manifest
from load_prediction.models import ForecastFrame, build_model
from load_prediction.pipeline.pipeline_output import save_forecast_csv
from load_prediction.pipeline.pipeline_result import PipelineRunResult
from load_prediction.pipeline.pipeline_splitting import split_train_test
from load_prediction.postprocessing import ForecastPostProcessor
from load_prediction.preprocessing import TimeSeriesCleaner
from load_prediction.visualization import plot_forecast_vs_actual, plot_training_curves

logger = logging.getLogger(__name__)


class LoadForecastPipeline:
    """End-to-end pipeline for one configured forecast scale."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cleaner = TimeSeriesCleaner(config.cleaning)
        self.model = build_model(config.model, config.scale)
        self.postprocessor = ForecastPostProcessor(config.postprocess)

    def run(
        self,
        data: TimeSeriesDataset,
        known_covariates: pd.DataFrame | None = None,
    ) -> PipelineRunResult:
        output_dir = self._output_dir()
        log_handler = self._attach_train_log(output_dir)
        try:
            logger.info(
                "Starting pipeline scale=%s model=%s input_rows=%s output_dir=%s",
                self.config.scale.name,
                self.config.model.name,
                len(data.frame),
                output_dir,
            )
            self._configure_model_output(output_dir / "model")
            scale_data = self._data_for_scale(data)
            cleaned = self.cleaner.fit_transform(scale_data)
            logger.info("Cleaning complete rows=%s", len(cleaned.frame))
            train_data, test_data = split_train_test(cleaned, self.config)
            logger.info(
                "Split complete train_rows=%s test_rows=%s strategy=%s train_ratio=%s holdout_length=%s",
                len(train_data.frame),
                len(test_data.frame),
                self.config.evaluation.split_strategy,
                self.config.evaluation.train_ratio,
                self.config.evaluation.holdout_length,
            )
            prediction_length = self.config.scale.prediction_length
            self._configure_prediction_length(prediction_length)
            self.model.fit(train_data)
            forecast = self._predict_test_windows(
                train_data,
                test_data,
                prediction_length,
                known_covariates=known_covariates,
            )
            residual_std = getattr(self.model, "residual_std_", 0.0)
            forecast = self.postprocessor.transform(forecast, residual_std=residual_std)
            metrics = evaluate_forecast(test_data.frame, forecast, target_col=train_data.target_col)
            logger.info("Pipeline finished metrics=%s", metrics)

            artifact_paths = self._save_run_outputs(output_dir, forecast, metrics, test_data)
            return PipelineRunResult(
                scale_name=self.config.scale.name,
                forecast=forecast,
                metrics=metrics,
                train_data=train_data,
                test_data=test_data,
                output_dir=output_dir,
                **artifact_paths,
            )
        finally:
            self._detach_train_log(log_handler)

    def fit_predict(
        self,
        data: TimeSeriesDataset,
        prediction_length: int | None = None,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        logger.info(
            "Starting fit_predict scale=%s model=%s input_rows=%s prediction_length=%s",
            self.config.scale.name,
            self.config.model.name,
            len(data.frame),
            prediction_length or self.config.scale.prediction_length,
        )
        scale_data = self._data_for_scale(data)
        cleaned = self.cleaner.fit_transform(scale_data)
        self.model.fit(cleaned)
        forecast = self.model.predict(
            cleaned,
            prediction_length=prediction_length or self.config.scale.prediction_length,
            freq=self.config.scale.freq,
            known_covariates=known_covariates,
        )
        residual_std = getattr(self.model, "residual_std_", 0.0)
        return self.postprocessor.transform(forecast, residual_std=residual_std)

    def _data_for_scale(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            frame=data.frame.copy(),
            timestamp_col=data.timestamp_col,
            target_col=data.target_col,
            item_id_col=data.item_id_col,
            freq=self.config.scale.freq,
        )

    def _prediction_length(self, test_data: TimeSeriesDataset) -> int:
        lengths = test_data.frame.groupby(test_data.item_id_col).size().unique()
        if len(lengths) != 1:
            raise ValueError("Each item must have the same test length for recursive prediction")
        return int(lengths[0])

    def _predict_test_windows(
        self,
        train_data: TimeSeriesDataset,
        test_data: TimeSeriesDataset,
        prediction_length: int,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        test_length = self._prediction_length(test_data)
        full_test_length = test_length - (test_length % prediction_length)
        if full_test_length <= 0:
            raise ValueError(
                f"Test set length={test_length} is shorter than prediction_length={prediction_length}"
            )
        if full_test_length != test_length:
            logger.warning(
                "Dropping trailing incomplete evaluation window rows=%s because prediction_length=%s",
                test_length - full_test_length,
                prediction_length,
            )
        forecast_parts: list[pd.DataFrame] = []
        selected_covariates = [
            col
            for col in self.config.scale.feature.known_covariates
            if col in test_data.covariate_columns
        ] or test_data.covariate_columns
        logger.info(
            "Running rolling evaluation windows test_length=%s prediction_length=%s windows=%s",
            full_test_length,
            prediction_length,
            full_test_length // prediction_length,
        )

        for window_index, start in enumerate(range(0, full_test_length, prediction_length), 1):
            end = start + prediction_length
            history_frame = self._history_for_window(train_data, test_data, end=start)
            window_frame = self._window_frame(test_data, start=start, end=end)
            prediction_covariates = known_covariates
            if prediction_covariates is None and selected_covariates:
                prediction_covariates = window_frame[
                    [test_data.timestamp_col, test_data.item_id_col, *selected_covariates]
                ]

            raw_forecast = self.model.predict(
                train_data.with_frame(history_frame),
                prediction_length=prediction_length,
                freq=self.config.scale.freq,
                known_covariates=prediction_covariates,
            )
            expected_keys = window_frame[
                [test_data.item_id_col, test_data.timestamp_col]
            ].drop_duplicates()
            aligned = raw_forecast.frame.merge(
                expected_keys,
                on=[test_data.item_id_col, test_data.timestamp_col],
                how="inner",
            )
            aligned["evaluation_window"] = window_index
            forecast_parts.append(aligned)
            logger.info(
                "Finished rolling prediction window index=%s start=%s end=%s rows=%s",
                window_index,
                start,
                end,
                len(aligned),
            )

        forecast_frame = pd.concat(forecast_parts, ignore_index=True)
        return ForecastFrame(
            frame=forecast_frame,
            timestamp_col=test_data.timestamp_col,
            item_id_col=test_data.item_id_col,
            prediction_col="prediction",
        )

    def _history_for_window(
        self,
        train_data: TimeSeriesDataset,
        test_data: TimeSeriesDataset,
        end: int,
    ) -> pd.DataFrame:
        history_parts = [train_data.frame]
        if end > 0:
            prior_parts = []
            for _, group in test_data.frame.groupby(test_data.item_id_col, sort=False):
                prior_parts.append(group.iloc[:end])
            history_parts.append(pd.concat(prior_parts, ignore_index=True))
        return pd.concat(history_parts, ignore_index=True)

    def _window_frame(
        self,
        test_data: TimeSeriesDataset,
        start: int,
        end: int,
    ) -> pd.DataFrame:
        window_parts = []
        for _, group in test_data.frame.groupby(test_data.item_id_col, sort=False):
            window_parts.append(group.iloc[start:end])
        return pd.concat(window_parts, ignore_index=True)

    def _configure_prediction_length(self, prediction_length: int) -> None:
        if hasattr(self.model, "prediction_length"):
            configured = getattr(self.model, "prediction_length")
            if configured != prediction_length:
                logger.info(
                    "Updating model prediction_length from %s to test length %s",
                    configured,
                    prediction_length,
                )
                setattr(self.model, "prediction_length", prediction_length)

    def _output_dir(self) -> Path:
        output = self.config.output
        base = Path(output.root_dir) / self.config.model.name / self.config.scale.name
        if output.run_name:
            base = base / output.run_name
        base.mkdir(parents=True, exist_ok=True)
        (base / "artifacts").mkdir(parents=True, exist_ok=True)
        (base / "logs").mkdir(parents=True, exist_ok=True)
        if output.save_model:
            (base / "model").mkdir(parents=True, exist_ok=True)
        return base

    def _attach_train_log(
        self,
        output_dir: Path,
    ) -> tuple[logging.Handler, int, int]:
        log_path = output_dir / "logs" / self.config.output.train_log_filename
        root_logger = logging.getLogger()
        package_logger = logging.getLogger("load_prediction")
        previous_root_level = root_logger.level
        previous_package_level = package_logger.level
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        if package_logger.level > logging.INFO or package_logger.level == logging.NOTSET:
            package_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
        root_logger.addHandler(handler)
        logger.info("Writing pipeline train log to %s", log_path)
        return handler, previous_root_level, previous_package_level

    def _detach_train_log(self, log_state: tuple[logging.Handler, int, int] | None) -> None:
        if log_state is None:
            return
        handler, previous_root_level, previous_package_level = log_state
        root_logger = logging.getLogger()
        package_logger = logging.getLogger("load_prediction")
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_root_level)
        package_logger.setLevel(previous_package_level)
        handler.close()

    def _configure_model_output(self, model_dir: Path) -> None:
        if hasattr(self.model, "path"):
            current_path = getattr(self.model, "path")
            if current_path and Path(str(current_path)) != model_dir:
                logger.warning(
                    "Overriding model output path from %s to pipeline-managed path %s",
                    current_path,
                    model_dir,
                )
            setattr(self.model, "path", str(model_dir))

    def _save_run_outputs(
        self,
        output_dir: Path,
        forecast: ForecastFrame,
        metrics: dict[str, float] | None,
        test_data: TimeSeriesDataset | None,
    ) -> dict[str, Path | None]:
        paths: dict[str, Path | None] = {
            "model_path": None,
            "forecast_csv_path": None,
            "forecast_plot_path": None,
            "metrics_path": None,
            "training_history_csv_path": None,
            "training_history_json_path": None,
            "training_curves_path": None,
        }
        if self.config.output.save_model:
            model_dir = output_dir / "model"
            paths["model_path"] = self.model.save(model_dir)
            save_model_manifest(build_model_manifest(self.config), model_dir)
            logger.info("Saved model artifact path=%s", paths["model_path"])
        if self.config.output.save_artifacts:
            paths["forecast_csv_path"] = output_dir / "artifacts" / self.config.output.forecast_filename
            save_forecast_csv(forecast, paths["forecast_csv_path"])
        if self.config.output.save_metrics and metrics is not None:
            metrics_path = output_dir / "artifacts" / self.config.output.metrics_filename
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            paths["metrics_path"] = metrics_path
            logger.info("Saved metrics JSON path=%s", metrics_path)
        if self.config.output.save_plot and test_data is not None:
            plot_path = output_dir / "artifacts" / self.config.output.plot_filename
            paths["forecast_plot_path"] = plot_forecast_vs_actual(
                test_data.frame,
                forecast,
                target_col=test_data.target_col,
                output_path=plot_path,
            )
        training_paths = self._save_training_outputs(output_dir)
        paths.update(training_paths)
        return paths

    def _save_training_outputs(self, output_dir: Path) -> dict[str, Path | None]:
        paths: dict[str, Path | None] = {
            "training_history_csv_path": None,
            "training_history_json_path": None,
            "training_curves_path": None,
        }
        training_log = self.model.training_log()
        history = training_log.get("history") if isinstance(training_log, dict) else None
        if not history:
            logger.info("Model did not provide structured training history; skipping training curve artifacts")
            return paths

        history_frame = pd.DataFrame(history)
        artifacts_dir = output_dir / "artifacts"
        csv_path = artifacts_dir / self.config.output.training_history_filename
        json_path = artifacts_dir / self.config.output.training_history_json_filename
        plot_path = artifacts_dir / self.config.output.training_curves_filename
        history_frame.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["training_history_csv_path"] = csv_path
        paths["training_history_json_path"] = json_path
        logger.info("Saved training history rows=%s path=%s", len(history_frame), csv_path)
        if self.config.output.save_plot:
            paths["training_curves_path"] = plot_training_curves(
                history_frame,
                output_path=plot_path,
                title=f"{self.config.model.name} Training Curves",
            )
        return paths
