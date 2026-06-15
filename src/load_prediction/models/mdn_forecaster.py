"""Mixture density network forecaster for point and probabilistic prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_prediction.configs import FeatureEngineeringConfig, ModelSpecConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features import FeatureBuilder
from load_prediction.models.base_forecaster import BaseForecastModel, ForecastFrame
from load_prediction.models.recursive_tabular import RecursiveTabularForecaster

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("MDN model requires PyTorch. Install with: pip install torch") from exc
    return torch, nn, DataLoader, TensorDataset


@dataclass
class MDNForecaster(BaseForecastModel):
    """Recursive tabular forecaster with a Gaussian mixture density network."""

    feature_builder: FeatureBuilder
    n_components: int = 3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    nll_loss_weight: float = 1.0
    mae_loss_weight: float = 0.0
    lr_scheduler_enabled: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    min_learning_rate: float = 1e-5
    train_progress_log_interval: int = 10
    validation_fraction: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    max_train_samples: int | None = 4096
    train_sample_strategy: str = "random"
    random_state: int = 42
    device: str = "cpu"
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    distribution_grid_size: int = 0
    distribution_std_width: float = 4.0
    min_std: float = 1e-3
    scale_features: bool = True
    scale_target: bool = True
    feature_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    model: object | None = None
    input_columns_: list[str] | None = None
    residual_std_: float = 0.0
    training_history_: list[dict[str, float | int | None]] | None = None
    fitted_: bool = False
    target_col_: str = "target"

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        feature_config: FeatureEngineeringConfig,
    ) -> "MDNForecaster":
        params = dict(model_config.params)
        return cls(
            feature_builder=FeatureBuilder(feature_config),
            n_components=int(params.pop("n_components", 3)),
            hidden_size=int(params.pop("hidden_size", 64)),
            num_layers=int(params.pop("num_layers", 2)),
            dropout=float(params.pop("dropout", 0.1)),
            epochs=int(params.pop("epochs", 20)),
            batch_size=int(params.pop("batch_size", 128)),
            learning_rate=float(params.pop("learning_rate", 1e-3)),
            weight_decay=float(params.pop("weight_decay", 0.0)),
            nll_loss_weight=float(params.pop("nll_loss_weight", 1.0)),
            mae_loss_weight=float(params.pop("mae_loss_weight", 0.0)),
            lr_scheduler_enabled=bool(params.pop("lr_scheduler_enabled", True)),
            lr_scheduler_factor=float(params.pop("lr_scheduler_factor", 0.5)),
            lr_scheduler_patience=int(params.pop("lr_scheduler_patience", 3)),
            min_learning_rate=float(params.pop("min_learning_rate", 1e-5)),
            train_progress_log_interval=int(params.pop("train_progress_log_interval", 10)),
            validation_fraction=float(params.pop("validation_fraction", 0.1)),
            early_stopping_patience=int(params.pop("early_stopping_patience", 5)),
            early_stopping_min_delta=float(params.pop("early_stopping_min_delta", 0.0)),
            max_train_samples=params.pop("max_train_samples", 4096),
            train_sample_strategy=str(params.pop("train_sample_strategy", "random")),
            random_state=model_config.random_state,
            device=str(params.pop("device", "cpu")),
            lower_quantile=float(params.pop("lower_quantile", 0.1)),
            upper_quantile=float(params.pop("upper_quantile", 0.9)),
            distribution_grid_size=int(params.pop("distribution_grid_size", 0)),
            distribution_std_width=float(params.pop("distribution_std_width", 4.0)),
            min_std=float(params.pop("min_std", 1e-3)),
            scale_features=bool(params.pop("scale_features", True)),
            scale_target=bool(params.pop("scale_target", True)),
        )

    def fit(self, data: TimeSeriesDataset) -> "MDNForecaster":
        torch, nn, DataLoader, TensorDataset = _import_torch()
        self._set_seed(torch)
        x_train, y_train = self.feature_builder.fit_transform(data)
        x_train, y_train = self._sample_training_rows(x_train, y_train)
        logger.info(
            "Fitting MDNForecaster rows=%s features=%s components=%s epochs=%s "
            "nll_loss_weight=%s mae_loss_weight=%s train_sample_strategy=%s "
            "lr_scheduler_enabled=%s lr_scheduler_factor=%s lr_scheduler_patience=%s",
            len(x_train),
            x_train.shape[1],
            self.n_components,
            self.epochs,
            self.nll_loss_weight,
            self.mae_loss_weight,
            self.train_sample_strategy,
            self.lr_scheduler_enabled,
            self.lr_scheduler_factor,
            self.lr_scheduler_patience,
        )
        self.input_columns_ = list(x_train.columns)
        x_model = self._fit_transform_features(x_train).to_numpy(dtype=np.float32)
        y_model = self._fit_transform_target(y_train).to_numpy(dtype=np.float32).reshape(-1, 1)
        train_slice, validation_slice = self._train_validation_slices(len(y_model))
        train_dataset = TensorDataset(
            torch.tensor(x_model[train_slice], dtype=torch.float32),
            torch.tensor(y_model[train_slice], dtype=torch.float32),
        )
        validation_loader = None
        if validation_slice is not None:
            validation_dataset = TensorDataset(
                torch.tensor(x_model[validation_slice], dtype=torch.float32),
                torch.tensor(y_model[validation_slice], dtype=torch.float32),
            )
            validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        self.model = _MDNNet(
            input_size=x_model.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_components=self.n_components,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = None
        if self.lr_scheduler_enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                threshold=self.early_stopping_min_delta,
                min_lr=self.min_learning_rate,
            )
        best_state = None
        best_validation_loss = float("inf")
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(1, self.epochs + 1):
            losses = []
            self.model.train()
            total_batches = len(train_loader)
            for batch_index, (features_batch, target_batch) in enumerate(train_loader, 1):
                features_batch = features_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                optimizer.zero_grad()
                weights, means, stds = self.model(features_batch)
                loss = self._loss(torch, target_batch, weights, means, stds)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                self._log_train_progress(
                    epoch=epoch,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    losses=losses,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )
            train_loss = float(np.mean(losses))
            validation_loss = None
            if validation_loader is not None:
                validation_loss = self._evaluate_loader_loss(torch, validation_loader)
                previous_learning_rate = float(optimizer.param_groups[0]["lr"])
                if scheduler is not None:
                    scheduler.step(validation_loss)
                current_learning_rate = float(optimizer.param_groups[0]["lr"])
                if current_learning_rate < previous_learning_rate:
                    logger.info(
                        "Adjusted MDN learning_rate epoch=%s from %.8f to %.8f validation_loss=%.6f",
                        epoch,
                        previous_learning_rate,
                        current_learning_rate,
                        validation_loss,
                    )
                if validation_loss + self.early_stopping_min_delta < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            else:
                previous_learning_rate = float(optimizer.param_groups[0]["lr"])
                if scheduler is not None:
                    scheduler.step(train_loss)
                current_learning_rate = float(optimizer.param_groups[0]["lr"])
                if current_learning_rate < previous_learning_rate:
                    logger.info(
                        "Adjusted MDN learning_rate epoch=%s from %.8f to %.8f train_loss=%.6f",
                        epoch,
                        previous_learning_rate,
                        current_learning_rate,
                        train_loss,
                    )
            self.training_history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            logger.info(
                "MDN epoch=%s train_loss=%.6f validation_loss=%s learning_rate=%.8f",
                epoch,
                train_loss,
                f"{validation_loss:.6f}" if validation_loss is not None else None,
                float(optimizer.param_groups[0]["lr"]),
            )
            if validation_loader is not None and stale_epochs >= self.early_stopping_patience:
                logger.info(
                    "MDN early stopping triggered epoch=%s best_validation_loss=%.6f patience=%s",
                    epoch,
                    best_validation_loss,
                    self.early_stopping_patience,
                )
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)

        with torch.no_grad():
            self.model.eval()
            sample_count = min(len(x_model), 512)
            weights, means, _ = self.model(torch.tensor(x_model[:sample_count], dtype=torch.float32).to(self.device))
            pred_scaled = torch.sum(weights * means, dim=1).detach().cpu().numpy()
        pred_target = self._inverse_transform_target(pred_scaled)
        actual_target = y_train.iloc[:sample_count].to_numpy(dtype=float)
        self.residual_std_ = float(np.nan_to_num(np.std(actual_target - pred_target, ddof=1), nan=0.0))
        self.fitted_ = True
        self.target_col_ = data.target_col
        logger.info("Finished fitting MDNForecaster residual_std=%.6f", self.residual_std_)
        return self

    def predict(
        self,
        history: TimeSeriesDataset,
        prediction_length: int,
        freq: str,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        if not self.fitted_ or self.model is None:
            raise RuntimeError("Model must be fitted before predict")
        torch, _, _, _ = _import_torch()
        history_frame = history.frame.copy()
        known_lookup = RecursiveTabularForecaster._build_known_covariate_lookup(
            self,
            known_covariates,
            history,
        )
        forecast_rows: list[dict[str, object]] = []
        self.model.eval()
        for item_id in history.item_ids:
            item_history = history_frame[history_frame[history.item_id_col] == item_id].copy()
            item_history = item_history.sort_values(history.timestamp_col)
            last_timestamp = pd.Timestamp(item_history[history.timestamp_col].max())
            future_index = pd.date_range(last_timestamp, periods=prediction_length + 1, freq=freq)[1:]
            for step, timestamp in enumerate(future_index, 1):
                known = known_lookup.get((str(item_id), pd.Timestamp(timestamp)), {})
                features = self.feature_builder.transform_future_row(
                    item_history,
                    history,
                    pd.Timestamp(timestamp),
                    str(item_id),
                    known,
                )
                model_features = self._transform_features(features)
                with torch.no_grad():
                    weights_t, means_t, stds_t = self.model(
                        torch.tensor(model_features.to_numpy(dtype=np.float32), dtype=torch.float32).to(self.device)
                    )
                weights = weights_t[0].detach().cpu().numpy().astype(float)
                means = self._inverse_transform_target(means_t[0].detach().cpu().numpy().astype(float))
                stds = self._inverse_transform_std(stds_t[0].detach().cpu().numpy().astype(float))
                prediction = float(np.sum(weights * means))
                row = self._distribution_row(weights, means, stds, prediction)
                row.update(
                    {
                        history.item_id_col: item_id,
                        history.timestamp_col: pd.Timestamp(timestamp),
                        "horizon_step": step,
                        "prediction": prediction,
                    }
                )
                forecast_rows.append(row)
                new_row = {
                    history.item_id_col: item_id,
                    history.timestamp_col: pd.Timestamp(timestamp),
                    history.target_col: prediction,
                }
                for covariate in history.covariate_columns:
                    new_row[covariate] = known.get(covariate, 0.0)
                item_history = pd.concat([item_history, pd.DataFrame([new_row])], ignore_index=True)
        return ForecastFrame(
            frame=pd.DataFrame(forecast_rows),
            timestamp_col=history.timestamp_col,
            item_id_col=history.item_id_col,
            prediction_col="prediction",
        )

    def training_log(self) -> dict[str, Any]:
        return {
            "history": list(self.training_history_ or []),
            "metadata": {
                "model_type": "mdn",
                "n_components": self.n_components,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "nll_loss_weight": self.nll_loss_weight,
                "mae_loss_weight": self.mae_loss_weight,
                "lr_scheduler_enabled": self.lr_scheduler_enabled,
                "lr_scheduler_factor": self.lr_scheduler_factor,
                "lr_scheduler_patience": self.lr_scheduler_patience,
                "min_learning_rate": self.min_learning_rate,
                "train_progress_log_interval": self.train_progress_log_interval,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "max_train_samples": self.max_train_samples,
                "train_sample_strategy": self.train_sample_strategy,
            },
        }

    def save(self, path: str | Path) -> Path:
        torch, _, _, _ = _import_torch()
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model is not None else None,
                "metadata": self._metadata(),
                "state": {
                    "feature_builder": self.feature_builder,
                    "feature_scaler": self.feature_scaler,
                    "target_scaler": self.target_scaler,
                    "input_columns": self.input_columns_,
                    "residual_std": self.residual_std_,
                    "training_history": self.training_history_,
                    "target_col": self.target_col_,
                },
            },
            model_path,
        )
        logger.info("Saved MDN checkpoint path=%s", model_path)
        return model_path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MDNForecaster":
        torch, _, _, _ = _import_torch()
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / "model.pt"
        try:
            checkpoint = torch.load(model_path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device or "cpu")
        metadata = checkpoint["metadata"]
        state = checkpoint["state"]
        forecaster = cls(
            feature_builder=state["feature_builder"],
            n_components=metadata["n_components"],
            hidden_size=metadata["hidden_size"],
            num_layers=metadata["num_layers"],
            dropout=metadata["dropout"],
            epochs=metadata["epochs"],
            batch_size=metadata["batch_size"],
            learning_rate=metadata["learning_rate"],
            weight_decay=metadata["weight_decay"],
            nll_loss_weight=metadata.get("nll_loss_weight", 1.0),
            mae_loss_weight=metadata.get("mae_loss_weight", 0.0),
            lr_scheduler_enabled=metadata.get("lr_scheduler_enabled", True),
            lr_scheduler_factor=metadata.get("lr_scheduler_factor", 0.5),
            lr_scheduler_patience=metadata.get("lr_scheduler_patience", 3),
            min_learning_rate=metadata.get("min_learning_rate", 1e-5),
            train_progress_log_interval=metadata.get("train_progress_log_interval", 10),
            validation_fraction=metadata["validation_fraction"],
            early_stopping_patience=metadata["early_stopping_patience"],
            early_stopping_min_delta=metadata["early_stopping_min_delta"],
            max_train_samples=metadata["max_train_samples"],
            train_sample_strategy=metadata.get("train_sample_strategy", "random"),
            random_state=metadata["random_state"],
            device=device or metadata["device"],
            lower_quantile=metadata["lower_quantile"],
            upper_quantile=metadata["upper_quantile"],
            distribution_grid_size=metadata["distribution_grid_size"],
            distribution_std_width=metadata["distribution_std_width"],
            min_std=metadata["min_std"],
            scale_features=metadata["scale_features"],
            scale_target=metadata["scale_target"],
        )
        forecaster.feature_scaler = state["feature_scaler"]
        forecaster.target_scaler = state["target_scaler"]
        forecaster.input_columns_ = state["input_columns"]
        forecaster.residual_std_ = float(state.get("residual_std", 0.0))
        forecaster.training_history_ = state.get("training_history") or []
        forecaster.target_col_ = state.get("target_col", "target")
        forecaster.model = _MDNNet(
            input_size=len(forecaster.input_columns_ or []),
            hidden_size=forecaster.hidden_size,
            num_layers=forecaster.num_layers,
            dropout=forecaster.dropout,
            n_components=forecaster.n_components,
        ).to(forecaster.device)
        forecaster.model.load_state_dict(checkpoint["state_dict"])
        forecaster.model.eval()
        forecaster.fitted_ = True
        logger.info("Loaded MDN checkpoint path=%s device=%s", model_path, forecaster.device)
        return forecaster

    def _metadata(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "nll_loss_weight": self.nll_loss_weight,
            "mae_loss_weight": self.mae_loss_weight,
            "lr_scheduler_enabled": self.lr_scheduler_enabled,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "min_learning_rate": self.min_learning_rate,
            "train_progress_log_interval": self.train_progress_log_interval,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "max_train_samples": self.max_train_samples,
            "train_sample_strategy": self.train_sample_strategy,
            "random_state": self.random_state,
            "device": self.device,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "distribution_grid_size": self.distribution_grid_size,
            "distribution_std_width": self.distribution_std_width,
            "min_std": self.min_std,
            "scale_features": self.scale_features,
            "scale_target": self.scale_target,
        }

    def _distribution_row(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        prediction: float,
    ) -> dict[str, object]:
        lower, upper = self._mixture_quantiles(weights, means, stds)
        row: dict[str, object] = {
            "0.1": float(lower),
            "0.9": float(upper),
            "prediction_lower": float(lower),
            "prediction_upper": float(upper),
            "prediction_density": float(self._mixture_pdf(np.array([prediction]), weights, means, stds)[0]),
            "mdn_weights": json.dumps(weights.tolist()),
            "mdn_means": json.dumps(means.tolist()),
            "mdn_stds": json.dumps(stds.tolist()),
        }
        for index, (weight, mean, std) in enumerate(zip(weights, means, stds), 1):
            row[f"mdn_weight_{index}"] = float(weight)
            row[f"mdn_mean_{index}"] = float(mean)
            row[f"mdn_std_{index}"] = float(std)
        if self.distribution_grid_size > 1:
            grid = self._distribution_grid(prediction, means, stds)
            row["distribution_values"] = json.dumps(grid.tolist())
            row["distribution_probabilities"] = json.dumps(
                self._mixture_pdf(grid, weights, means, stds).tolist()
            )
        return row

    def _fit_transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_features:
            return features
        self.feature_scaler = StandardScaler()
        values = self.feature_scaler.fit_transform(features)
        return pd.DataFrame(values, columns=features.columns, index=features.index)

    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.scale_features:
            if self.feature_scaler is None:
                raise RuntimeError("Feature scaler has not been fitted")
            values = self.feature_scaler.transform(features)
            return pd.DataFrame(values, columns=features.columns, index=features.index)
        return features

    def _fit_transform_target(self, target: pd.Series) -> pd.Series:
        if not self.scale_target:
            return target
        self.target_scaler = StandardScaler()
        values = self.target_scaler.fit_transform(target.to_numpy(dtype=float).reshape(-1, 1)).ravel()
        return pd.Series(values, index=target.index)

    def _inverse_transform_target(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(-1, 1)
        if self.scale_target:
            if self.target_scaler is None:
                raise RuntimeError("Target scaler has not been fitted")
            return self.target_scaler.inverse_transform(array).ravel()
        return array.ravel()

    def _inverse_transform_std(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if self.scale_target and self.target_scaler is not None:
            scale = float(self.target_scaler.scale_[0])
            return np.maximum(array * scale, self.min_std)
        return np.maximum(array, self.min_std)

    def _train_validation_slices(self, sample_count: int) -> tuple[slice, slice | None]:
        if self.early_stopping_patience <= 0 or self.validation_fraction <= 0 or sample_count < 2:
            return slice(None), None
        validation_count = int(round(sample_count * self.validation_fraction))
        validation_count = max(1, min(validation_count, sample_count - 1))
        train_count = sample_count - validation_count
        return slice(0, train_count), slice(train_count, sample_count)

    def _sample_training_rows(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.max_train_samples is None:
            return features, target
        max_samples = int(self.max_train_samples)
        if max_samples <= 0:
            raise ValueError("max_train_samples must be positive when provided")
        if len(features) <= max_samples:
            return features, target

        strategy = self.train_sample_strategy.lower()
        if strategy == "random":
            rng = np.random.default_rng(self.random_state)
            indices = np.sort(rng.choice(len(features), size=max_samples, replace=False))
        elif strategy == "first":
            indices = np.arange(max_samples)
        elif strategy == "last":
            indices = np.arange(len(features) - max_samples, len(features))
        elif strategy in {"uniform_time", "uniform"}:
            indices = np.linspace(0, len(features) - 1, max_samples, dtype=int)
        else:
            raise ValueError(
                "Unsupported train_sample_strategy "
                f"'{self.train_sample_strategy}'. Use random, first, last, or uniform_time."
            )
        return features.iloc[indices], target.iloc[indices]

    def _log_train_progress(
        self,
        epoch: int,
        batch_index: int,
        total_batches: int,
        losses: list[float],
        learning_rate: float,
    ) -> None:
        interval = max(1, int(self.train_progress_log_interval))
        if batch_index != 1 and batch_index != total_batches and batch_index % interval != 0:
            return
        progress_pct = 100.0 * batch_index / max(total_batches, 1)
        logger.info(
            "MDN training progress epoch=%s/%s batch=%s/%s progress=%.1f%% batch_loss=%.6f avg_loss=%.6f learning_rate=%.8f",
            epoch,
            self.epochs,
            batch_index,
            total_batches,
            progress_pct,
            losses[-1],
            float(np.mean(losses)),
            learning_rate,
        )

    def _evaluate_loader_loss(self, torch: Any, loader: Any) -> float:
        losses = []
        self.model.eval()
        with torch.no_grad():
            for features_batch, target_batch in loader:
                weights, means, stds = self.model(features_batch.to(self.device))
                losses.append(
                    float(self._loss(torch, target_batch.to(self.device), weights, means, stds).detach().cpu())
                )
        self.model.train()
        return float(np.mean(losses))

    def _loss(self, torch: Any, target: Any, weights: Any, means: Any, stds: Any) -> Any:
        nll_loss = self._nll(torch, target, weights, means, stds)
        if self.mae_loss_weight <= 0:
            return self.nll_loss_weight * nll_loss
        point_prediction = torch.sum(weights * means, dim=1, keepdim=True)
        mae_loss = torch.mean(torch.abs(target - point_prediction))
        return self.nll_loss_weight * nll_loss + self.mae_loss_weight * mae_loss

    def _nll(self, torch: Any, target: Any, weights: Any, means: Any, stds: Any) -> Any:
        target = target.expand_as(means)
        log_component = -0.5 * ((target - means) / stds) ** 2 - torch.log(stds) - 0.5 * np.log(2 * np.pi)
        return -torch.logsumexp(torch.log(weights + 1e-12) + log_component, dim=1).mean()

    def _mixture_quantiles(self, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
        quantiles = self._mixture_inverse_cdf(
            weights,
            means,
            stds,
            np.array([self.lower_quantile, self.upper_quantile], dtype=float),
        )
        return max(0.0, float(quantiles[0])), float(quantiles[1])

    def _distribution_grid(self, prediction: float, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        low = np.min(means - self.distribution_std_width * stds)
        high = np.max(means + self.distribution_std_width * stds)
        low = min(low, prediction)
        high = max(high, prediction)
        return np.linspace(max(0.0, low), high, self.distribution_grid_size)

    def _mixture_inverse_cdf(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        low = np.min(means - 8.0 * stds)
        high = np.max(means + 8.0 * stds)
        xs = np.linspace(low, high, 2048)
        cdf = self._mixture_cdf(xs, weights, means, stds)
        return np.interp(probabilities, cdf, xs)

    def _mixture_pdf(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        means = means.reshape(1, -1)
        stds = np.maximum(stds.reshape(1, -1), self.min_std)
        z = (values - means) / stds
        component_pdf = np.exp(-0.5 * z**2) / (stds * np.sqrt(2 * np.pi))
        return np.sum(component_pdf * weights.reshape(1, -1), axis=1)

    def _mixture_cdf(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        means = means.reshape(1, -1)
        stds = np.maximum(stds.reshape(1, -1), self.min_std)
        z = (values - means) / (stds * np.sqrt(2.0))
        component_cdf = 0.5 * (1.0 + np.vectorize(_erf)(z))
        return np.sum(component_cdf * weights.reshape(1, -1), axis=1)

    def _set_seed(self, torch: Any) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)


class _MDNNet:
    def __new__(
        cls,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_components: int,
    ):
        torch, nn, _, _ = _import_torch()

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[Any] = []
                current_size = input_size
                for _ in range(max(1, num_layers)):
                    layers.append(nn.Linear(current_size, hidden_size))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    current_size = hidden_size
                self.backbone = nn.Sequential(*layers)
                self.weight_head = nn.Linear(current_size, n_components)
                self.mean_head = nn.Linear(current_size, n_components)
                self.std_head = nn.Linear(current_size, n_components)

            def forward(self, features):
                hidden = self.backbone(features)
                weights = torch.softmax(self.weight_head(hidden), dim=-1)
                means = self.mean_head(hidden)
                stds = torch.nn.functional.softplus(self.std_head(hidden)) + 1e-3
                return weights, means, stds

        return Net()


def _erf(value: float) -> float:
    import math

    return math.erf(float(value))
