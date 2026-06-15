"""PyTorch LSTM forecaster with mixture density output."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_prediction.configs import ModelSpecConfig, ForecastProfileConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.models.base_forecaster import BaseForecastModel, ForecastFrame

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("LSTM model requires PyTorch. Install with: pip install torch") from exc
    return torch, nn, DataLoader, TensorDataset


class _LSTMMDNNet:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(
        history_numeric_size: int,
        future_numeric_size: int,
        categorical_cardinalities: list[int],
        embedding_dims: list[int],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_components: int,
        min_std: float,
        bidirectional: bool = False,
        use_point_head: bool = False,
    ):
        torch, nn, _, _ = _import_torch()

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if len(categorical_cardinalities) != len(embedding_dims):
                    raise ValueError("categorical_cardinalities and embedding_dims must have the same length")
                self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(cardinality + 1, embedding_dim)
                        for cardinality, embedding_dim in zip(categorical_cardinalities, embedding_dims)
                    ]
                )
                embedded_size = int(sum(embedding_dims))
                input_size = history_numeric_size + embedded_size
                future_size = future_numeric_size + embedded_size
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.encoder = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=lstm_dropout,
                    bidirectional=bidirectional,
                )
                context_size = hidden_size * 2 if bidirectional else hidden_size
                self.head = nn.Sequential(
                    nn.Linear(context_size + future_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, n_components * 3),
                )
                self.point_head = None
                if use_point_head:
                    self.point_head = nn.Sequential(
                        nn.Linear(context_size + future_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, 1),
                    )
                self.n_components = n_components
                self.min_std = min_std

            def _concat_features(self, numeric, categorical):
                if not self.embeddings:
                    return numeric
                embedded = [
                    embedding(categorical[..., idx].long())
                    for idx, embedding in enumerate(self.embeddings)
                ]
                return torch.cat([numeric, *embedded], dim=-1)

            def forward(self, history_numeric, history_categorical, future_numeric, future_categorical):
                history = self._concat_features(history_numeric, history_categorical)
                future = self._concat_features(future_numeric, future_categorical)
                _, (hidden, _) = self.encoder(history)
                if bidirectional:
                    context = torch.cat([hidden[-2], hidden[-1]], dim=-1)
                else:
                    context = hidden[-1]
                horizon = future.shape[1]
                context = context.unsqueeze(1).expand(-1, horizon, -1)
                decoder_features = torch.cat([context, future], dim=-1)
                raw = self.head(decoder_features)
                raw = raw.reshape(*raw.shape[:-1], self.n_components, 3)
                weights = torch.nn.functional.softmax(raw[..., 0], dim=-1)
                means = raw[..., 1]
                stds = torch.nn.functional.softplus(raw[..., 2]) + self.min_std
                point = None
                if self.point_head is not None:
                    point = self.point_head(decoder_features).squeeze(-1)
                return weights, means, stds, point

        return Net()


@dataclass
class LSTMForecaster(BaseForecastModel):
    prediction_length: int
    freq: str
    known_covariates_names: list[str]
    random_state: int = 42
    context_length: int = 672
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler_enabled: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    min_learning_rate: float = 1e-5
    train_progress_log_interval: int = 10
    max_train_samples: int | None = 4096
    train_sample_strategy: str = "random"
    device: str = "cpu"
    bidirectional: bool = False
    embedding_dim: int | None = None
    validation_fraction: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    n_components: int = 3
    min_std: float = 1e-3
    use_point_head: bool = False
    nll_loss_weight: float = 1.0
    point_loss_weight: float = 0.0
    add_calendar_features: bool = True
    add_lag_features: bool = True
    lag_feature_steps: tuple[int, ...] = (96, 192, 672)
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    distribution_grid_size: int = 80
    distribution_std_width: float = 4.0
    model: object | None = None
    target_scaler: StandardScaler | None = None
    covariate_scaler: StandardScaler | None = None
    engineered_feature_scaler: StandardScaler | None = None
    numeric_covariates_: list[str] | None = None
    categorical_covariates_: list[str] | None = None
    category_maps_: dict[str, dict[str, int]] | None = None
    input_columns_: list[str] | None = None
    future_columns_: list[str] | None = None
    history_numeric_columns_: list[str] | None = None
    future_numeric_columns_: list[str] | None = None
    categorical_code_columns_: list[str] | None = None
    categorical_cardinalities_: list[int] | None = None
    embedding_dims_: list[int] | None = None
    training_history_: list[dict[str, float | int | None]] | None = None
    residual_std_: float = 0.0
    fitted_: bool = False

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        scale_config: ForecastProfileConfig,
    ) -> "LSTMForecaster":
        params = dict(model_config.params)
        return cls(
            prediction_length=scale_config.prediction_length,
            freq=scale_config.freq,
            known_covariates_names=list(scale_config.feature.known_covariates),
            random_state=model_config.random_state,
            context_length=params.pop("context_length", 672),
            hidden_size=params.pop("hidden_size", 64),
            num_layers=params.pop("num_layers", 1),
            dropout=params.pop("dropout", 0.1),
            epochs=params.pop("epochs", 5),
            batch_size=params.pop("batch_size", 64),
            learning_rate=params.pop("learning_rate", 1e-3),
            weight_decay=params.pop("weight_decay", 0.0),
            lr_scheduler_enabled=params.pop("lr_scheduler_enabled", True),
            lr_scheduler_factor=params.pop("lr_scheduler_factor", 0.5),
            lr_scheduler_patience=params.pop("lr_scheduler_patience", 3),
            min_learning_rate=params.pop("min_learning_rate", 1e-5),
            train_progress_log_interval=params.pop("train_progress_log_interval", 10),
            max_train_samples=params.pop("max_train_samples", 4096),
            train_sample_strategy=params.pop("train_sample_strategy", "random"),
            device=params.pop("device", "cpu"),
            bidirectional=params.pop("bidirectional", model_config.name in {"bilstm", "bi_lstm"}),
            embedding_dim=params.pop("embedding_dim", None),
            validation_fraction=params.pop("validation_fraction", 0.1),
            early_stopping_patience=params.pop("early_stopping_patience", 5),
            early_stopping_min_delta=params.pop("early_stopping_min_delta", 0.0),
            n_components=params.pop("n_components", 3),
            min_std=params.pop("min_std", 1e-3),
            use_point_head=params.pop("use_point_head", False),
            nll_loss_weight=params.pop("nll_loss_weight", 1.0),
            point_loss_weight=params.pop("point_loss_weight", 0.0),
            add_calendar_features=params.pop("add_calendar_features", True),
            add_lag_features=params.pop("add_lag_features", True),
            lag_feature_steps=tuple(params.pop("lag_feature_steps", (96, 192, 672))),
            lower_quantile=params.pop("lower_quantile", 0.1),
            upper_quantile=params.pop("upper_quantile", 0.9),
            distribution_grid_size=params.pop("distribution_grid_size", 80),
            distribution_std_width=params.pop("distribution_std_width", 4.0),
        )

    def fit(self, data: TimeSeriesDataset) -> "LSTMForecaster":
        torch, nn, DataLoader, TensorDataset = _import_torch()
        self._set_seed(torch)
        frame = data.frame.sort_values([data.item_id_col, data.timestamp_col]).copy()
        self._fit_feature_metadata(frame, data)
        prepared = self._prepare_frame(frame, data, fit_scalers=True)
        x_hist_num, x_hist_cat, x_future_num, x_future_cat, y = self._build_training_arrays(prepared, data)
        train_slice, validation_slice = self._train_validation_slices(len(y))
        train_dataset = TensorDataset(
            torch.tensor(x_hist_num[train_slice], dtype=torch.float32),
            torch.tensor(x_hist_cat[train_slice], dtype=torch.long),
            torch.tensor(x_future_num[train_slice], dtype=torch.float32),
            torch.tensor(x_future_cat[train_slice], dtype=torch.long),
            torch.tensor(y[train_slice], dtype=torch.float32),
        )
        validation_dataset = None
        if validation_slice is not None:
            validation_dataset = TensorDataset(
                torch.tensor(x_hist_num[validation_slice], dtype=torch.float32),
                torch.tensor(x_hist_cat[validation_slice], dtype=torch.long),
                torch.tensor(x_future_num[validation_slice], dtype=torch.float32),
                torch.tensor(x_future_cat[validation_slice], dtype=torch.long),
                torch.tensor(y[validation_slice], dtype=torch.float32),
            )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = None
        if validation_dataset is not None:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        self.model = _LSTMMDNNet.build(
            history_numeric_size=x_hist_num.shape[-1],
            future_numeric_size=x_future_num.shape[-1],
            categorical_cardinalities=self.categorical_cardinalities_ or [],
            embedding_dims=self.embedding_dims_ or [],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_components=self.n_components,
            min_std=self.min_std,
            bidirectional=self.bidirectional,
            use_point_head=self.use_point_head,
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
        logger.info(
            "Fitting %s samples=%s validation_samples=%s context_length=%s prediction_length=%s "
            "history_numeric_size=%s future_numeric_size=%s categorical_features=%s embedding_dims=%s "
            "epochs=%s train_sample_strategy=%s use_point_head=%s nll_loss_weight=%s point_loss_weight=%s "
            "lr_scheduler_enabled=%s lr_scheduler_factor=%s lr_scheduler_patience=%s",
            "BiLSTM" if self.bidirectional else "LSTM",
            len(train_dataset),
            len(validation_dataset) if validation_dataset is not None else 0,
            self.context_length,
            self.prediction_length,
            x_hist_num.shape[-1],
            x_future_num.shape[-1],
            self.categorical_covariates_,
            self.embedding_dims_,
            self.epochs,
            self.train_sample_strategy,
            self.use_point_head,
            self.nll_loss_weight,
            self.point_loss_weight,
            self.lr_scheduler_enabled,
            self.lr_scheduler_factor,
            self.lr_scheduler_patience,
        )
        self.model.train()
        best_state = None
        best_validation_loss = float("inf")
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(1, self.epochs + 1):
            losses = []
            total_batches = len(loader)
            for batch_index, (
                hist_num_batch,
                hist_cat_batch,
                future_num_batch,
                future_cat_batch,
                target_batch,
            ) in enumerate(loader, 1):
                hist_num_batch = hist_num_batch.to(self.device)
                hist_cat_batch = hist_cat_batch.to(self.device)
                future_num_batch = future_num_batch.to(self.device)
                future_cat_batch = future_cat_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                optimizer.zero_grad()
                weights, means, stds, point = self.model(
                    hist_num_batch,
                    hist_cat_batch,
                    future_num_batch,
                    future_cat_batch,
                )
                loss = self._loss(torch, target_batch, weights, means, stds, point)
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
                        "Adjusted LSTM learning_rate epoch=%s from %.8f to %.8f validation_loss=%.6f",
                        epoch,
                        previous_learning_rate,
                        current_learning_rate,
                        validation_loss,
                    )
                self.training_history_.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "learning_rate": current_learning_rate,
                    }
                )
                logger.info(
                    "LSTM epoch=%s train_loss=%.6f validation_loss=%.6f learning_rate=%.8f",
                    epoch,
                    train_loss,
                    validation_loss,
                    current_learning_rate,
                )
                if validation_loss + self.early_stopping_min_delta < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.early_stopping_patience:
                        logger.info(
                            "Early stopping triggered epoch=%s best_validation_loss=%.6f patience=%s",
                            epoch,
                            best_validation_loss,
                            self.early_stopping_patience,
                        )
                        break
            else:
                previous_learning_rate = float(optimizer.param_groups[0]["lr"])
                if scheduler is not None:
                    scheduler.step(train_loss)
                current_learning_rate = float(optimizer.param_groups[0]["lr"])
                if current_learning_rate < previous_learning_rate:
                    logger.info(
                        "Adjusted LSTM learning_rate epoch=%s from %.8f to %.8f train_loss=%.6f",
                        epoch,
                        previous_learning_rate,
                        current_learning_rate,
                        train_loss,
                    )
                self.training_history_.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": None,
                        "learning_rate": current_learning_rate,
                    }
                )
                logger.info(
                    "LSTM epoch=%s train_loss=%.6f learning_rate=%.8f",
                    epoch,
                    train_loss,
                    current_learning_rate,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best LSTM checkpoint validation_loss=%.6f", best_validation_loss)

        with torch.no_grad():
            self.model.eval()
            sample_count = min(len(y), 512)
            weights, means, _, point = self.model(
                torch.tensor(x_hist_num[:sample_count], dtype=torch.float32).to(self.device),
                torch.tensor(x_hist_cat[:sample_count], dtype=torch.long).to(self.device),
                torch.tensor(x_future_num[:sample_count], dtype=torch.float32).to(self.device),
                torch.tensor(x_future_cat[:sample_count], dtype=torch.long).to(self.device),
            )
            pred_tensor = point if point is not None else torch.sum(weights * means, dim=-1)
            pred = pred_tensor.detach().cpu().numpy()
        pred_target = self._inverse_target(pred.reshape(-1))
        actual_target = self._inverse_target(y[:sample_count].reshape(-1))
        self.residual_std_ = float(np.nan_to_num(np.std(actual_target - pred_target, ddof=1), nan=0.0))
        self.fitted_ = True
        logger.info("Finished fitting LSTM residual_std=%.6f", self.residual_std_)
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
        if prediction_length != self.prediction_length:
            raise ValueError(f"LSTM was trained for prediction_length={self.prediction_length}, got {prediction_length}")
        torch, _, _, _ = _import_torch()
        rows: list[dict[str, object]] = []
        self.model.eval()
        with torch.no_grad():
            for item_id in history.item_ids:
                item_history = history.frame[history.frame[history.item_id_col] == item_id].copy()
                item_history = item_history.sort_values(history.timestamp_col)
                prepared_history = self._prepare_frame(item_history, history, fit_scalers=False)
                if len(prepared_history) < self.context_length:
                    raise ValueError(
                        f"Not enough history for LSTM context_length={self.context_length}; got {len(prepared_history)}"
                    )
                future_frame = self._future_frame(item_history, history, item_id, freq, known_covariates)
                prepared_future = self._prepare_frame(future_frame, history, fit_scalers=False, include_target=False)
                hist_num = prepared_history[self.history_numeric_columns_].tail(self.context_length).to_numpy(dtype=float)
                hist_cat = prepared_history[self.categorical_code_columns_].tail(self.context_length).to_numpy(dtype=int)
                future_num = prepared_future[self.future_numeric_columns_].to_numpy(dtype=float)
                future_cat = prepared_future[self.categorical_code_columns_].to_numpy(dtype=int)
                weights_scaled, means_scaled, stds_scaled, point_scaled = self.model(
                    torch.tensor(hist_num[None, ...], dtype=torch.float32).to(self.device),
                    torch.tensor(hist_cat[None, ...], dtype=torch.long).to(self.device),
                    torch.tensor(future_num[None, ...], dtype=torch.float32).to(self.device),
                    torch.tensor(future_cat[None, ...], dtype=torch.long).to(self.device),
                )
                weights = weights_scaled[0].detach().cpu().numpy()
                means = self._inverse_target(means_scaled[0].detach().cpu().numpy().reshape(-1)).reshape(
                    self.prediction_length,
                    self.n_components,
                )
                stds = self._inverse_target_stds(stds_scaled[0].detach().cpu().numpy())
                if point_scaled is not None:
                    prediction = self._inverse_target(point_scaled[0].detach().cpu().numpy())
                else:
                    prediction = np.sum(weights * means, axis=1)
                for step, (_, future_row) in enumerate(future_frame.iterrows(), 1):
                    pred_value = float(prediction[step - 1])
                    row = self._distribution_row(
                        weights=weights[step - 1],
                        means=means[step - 1],
                        stds=stds[step - 1],
                        prediction=pred_value,
                    )
                    row.update(
                        {
                            history.item_id_col: item_id,
                            history.timestamp_col: pd.Timestamp(future_row[history.timestamp_col]),
                            "horizon_step": step,
                            "prediction": pred_value,
                        }
                    )
                    rows.append(row)
        return ForecastFrame(
            frame=pd.DataFrame(rows),
            timestamp_col=history.timestamp_col,
            item_id_col=history.item_id_col,
            prediction_col="prediction",
        )

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
                    "target_scaler": self.target_scaler,
                    "covariate_scaler": self.covariate_scaler,
                    "engineered_feature_scaler": self.engineered_feature_scaler,
                    "numeric_covariates": self.numeric_covariates_,
                    "categorical_covariates": self.categorical_covariates_,
                    "category_maps": self.category_maps_,
                    "input_columns": self.input_columns_,
                    "future_columns": self.future_columns_,
                    "history_numeric_columns": self.history_numeric_columns_,
                    "future_numeric_columns": self.future_numeric_columns_,
                    "categorical_code_columns": self.categorical_code_columns_,
                    "categorical_cardinalities": self.categorical_cardinalities_,
                    "embedding_dims": self.embedding_dims_,
                    "training_history": self.training_history_,
                    "residual_std": self.residual_std_,
                },
            },
            model_path,
        )
        logger.info("Saved LSTM checkpoint path=%s", model_path)
        return model_path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "LSTMForecaster":
        torch, _, _, _ = _import_torch()
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / "model.pt"
        try:
            checkpoint = torch.load(model_path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device or "cpu")
        metadata = checkpoint.get("metadata") or checkpoint.get("config") or {}
        state = checkpoint.get("state") or {}
        saved_history_numeric_columns = state.get("history_numeric_columns") or metadata.get("history_numeric_columns") or []
        saved_future_numeric_columns = state.get("future_numeric_columns") or metadata.get("future_numeric_columns") or []
        saved_numeric_columns = [*saved_history_numeric_columns, *saved_future_numeric_columns]
        has_saved_engineered_features = any(
            str(column).startswith("feature__calendar__")
            or str(column).startswith("feature__time__")
            or str(column).startswith("feature__lag__")
            for column in saved_numeric_columns
        )
        add_calendar_features = metadata.get("add_calendar_features")
        if add_calendar_features is None:
            add_calendar_features = any(
                str(column).startswith("feature__calendar__") or str(column).startswith("feature__time__")
                for column in saved_numeric_columns
            )
        add_lag_features = metadata.get("add_lag_features")
        if add_lag_features is None:
            add_lag_features = any(str(column).startswith("feature__lag__") for column in saved_numeric_columns)
        lag_feature_steps = tuple(metadata.get("lag_feature_steps", (96, 192, 672)))
        forecaster = cls(
            prediction_length=metadata["prediction_length"],
            freq=metadata["freq"],
            known_covariates_names=list(metadata.get("known_covariates_names", [])),
            random_state=metadata.get("random_state", 42),
            context_length=metadata.get("context_length", 672),
            hidden_size=metadata.get("hidden_size", 64),
            num_layers=metadata.get("num_layers", 1),
            dropout=metadata.get("dropout", 0.1),
            epochs=metadata.get("epochs", 5),
            batch_size=metadata.get("batch_size", 64),
            learning_rate=metadata.get("learning_rate", 1e-3),
            weight_decay=metadata.get("weight_decay", 0.0),
            lr_scheduler_enabled=metadata.get("lr_scheduler_enabled", True),
            lr_scheduler_factor=metadata.get("lr_scheduler_factor", 0.5),
            lr_scheduler_patience=metadata.get("lr_scheduler_patience", 3),
            min_learning_rate=metadata.get("min_learning_rate", 1e-5),
            train_progress_log_interval=metadata.get("train_progress_log_interval", 10),
            max_train_samples=metadata.get("max_train_samples", 4096),
            train_sample_strategy=metadata.get("train_sample_strategy", "random"),
            device=device or metadata.get("device", "cpu"),
            bidirectional=metadata.get("bidirectional", False),
            embedding_dim=metadata.get("embedding_dim"),
            validation_fraction=metadata.get("validation_fraction", 0.1),
            early_stopping_patience=metadata.get("early_stopping_patience", 5),
            early_stopping_min_delta=metadata.get("early_stopping_min_delta", 0.0),
            n_components=metadata.get("n_components", 3),
            min_std=metadata.get("min_std", 1e-3),
            use_point_head=metadata.get("use_point_head", False),
            nll_loss_weight=metadata.get("nll_loss_weight", 1.0),
            point_loss_weight=metadata.get("point_loss_weight", 0.0),
            add_calendar_features=bool(add_calendar_features),
            add_lag_features=bool(add_lag_features),
            lag_feature_steps=lag_feature_steps,
            lower_quantile=metadata.get("lower_quantile", 0.1),
            upper_quantile=metadata.get("upper_quantile", 0.9),
            distribution_grid_size=metadata.get("distribution_grid_size", 80),
            distribution_std_width=metadata.get("distribution_std_width", 4.0),
        )
        forecaster.target_scaler = state.get("target_scaler")
        forecaster.covariate_scaler = state.get("covariate_scaler")
        forecaster.engineered_feature_scaler = state.get("engineered_feature_scaler")
        if has_saved_engineered_features and forecaster.engineered_feature_scaler is None:
            logger.warning(
                "Loaded LSTM checkpoint has engineered feature columns but no engineered_feature_scaler; "
                "online prediction may be incompatible with this artifact."
            )
        forecaster.numeric_covariates_ = state.get("numeric_covariates") or []
        forecaster.categorical_covariates_ = state.get("categorical_covariates") or []
        forecaster.category_maps_ = state.get("category_maps") or {}
        forecaster.input_columns_ = state.get("input_columns") or metadata.get("input_columns") or []
        forecaster.future_columns_ = state.get("future_columns") or metadata.get("future_columns") or []
        forecaster.history_numeric_columns_ = state.get("history_numeric_columns") or metadata.get(
            "history_numeric_columns"
        ) or ["target_scaled", *forecaster.numeric_covariates_]
        forecaster.future_numeric_columns_ = state.get("future_numeric_columns") or metadata.get(
            "future_numeric_columns"
        ) or list(forecaster.numeric_covariates_)
        forecaster.categorical_code_columns_ = state.get("categorical_code_columns") or metadata.get(
            "categorical_code_columns"
        ) or [f"{col}__code" for col in forecaster.categorical_covariates_]
        forecaster.categorical_cardinalities_ = state.get("categorical_cardinalities") or metadata.get(
            "categorical_cardinalities"
        ) or [len(forecaster.category_maps_.get(col, {})) for col in forecaster.categorical_covariates_]
        forecaster.embedding_dims_ = state.get("embedding_dims") or metadata.get("embedding_dims") or [
            forecaster._embedding_size(cardinality) for cardinality in forecaster.categorical_cardinalities_
        ]
        forecaster.training_history_ = state.get("training_history") or metadata.get("training_history") or []
        forecaster.residual_std_ = float(state.get("residual_std", metadata.get("residual_std", 0.0)))
        forecaster.model = _LSTMMDNNet.build(
            history_numeric_size=len(forecaster.history_numeric_columns_),
            future_numeric_size=len(forecaster.future_numeric_columns_),
            categorical_cardinalities=forecaster.categorical_cardinalities_ or [],
            embedding_dims=forecaster.embedding_dims_ or [],
            hidden_size=forecaster.hidden_size,
            num_layers=forecaster.num_layers,
            dropout=forecaster.dropout,
            n_components=forecaster.n_components,
            min_std=forecaster.min_std,
            bidirectional=forecaster.bidirectional,
            use_point_head=forecaster.use_point_head,
        ).to(forecaster.device)
        forecaster.model.load_state_dict(checkpoint["state_dict"])
        forecaster.model.eval()
        forecaster.fitted_ = True
        logger.info("Loaded LSTM checkpoint path=%s device=%s", model_path, forecaster.device)
        return forecaster

    def training_log(self) -> dict[str, Any]:
        return {
            "history": list(self.training_history_ or []),
            "metadata": {
                "model_type": "bilstm" if self.bidirectional else "lstm",
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "lr_scheduler_enabled": self.lr_scheduler_enabled,
                "lr_scheduler_factor": self.lr_scheduler_factor,
                "lr_scheduler_patience": self.lr_scheduler_patience,
                "min_learning_rate": self.min_learning_rate,
                "train_progress_log_interval": self.train_progress_log_interval,
                "max_train_samples": self.max_train_samples,
                "train_sample_strategy": self.train_sample_strategy,
                "early_stopping_patience": self.early_stopping_patience,
                "validation_fraction": self.validation_fraction,
                "n_components": self.n_components,
                "min_std": self.min_std,
                "use_point_head": self.use_point_head,
                "nll_loss_weight": self.nll_loss_weight,
                "point_loss_weight": self.point_loss_weight,
                "add_calendar_features": self.add_calendar_features,
                "add_lag_features": self.add_lag_features,
                "lag_feature_steps": self.lag_feature_steps,
                "history_numeric_columns": self.history_numeric_columns_,
                "future_numeric_columns": self.future_numeric_columns_,
                "numeric_covariates": self.numeric_covariates_,
                "categorical_covariates": self.categorical_covariates_,
                "embedding_dims": self.embedding_dims_,
            },
        }

    def _set_seed(self, torch: Any) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _fit_feature_metadata(self, frame: pd.DataFrame, data: TimeSeriesDataset) -> None:
        covariates = [col for col in self.known_covariates_names if col in frame.columns]
        numeric = [col for col in covariates if pd.api.types.is_numeric_dtype(frame[col])]
        categorical = [col for col in covariates if col not in numeric]
        self.numeric_covariates_ = numeric
        self.categorical_covariates_ = categorical
        self.category_maps_ = {
            col: {str(value): idx + 1 for idx, value in enumerate(sorted(frame[col].dropna().astype(str).unique()))}
            for col in categorical
        }
        calendar_columns = self._calendar_feature_columns() if self.add_calendar_features else []
        lag_columns = self._lag_feature_columns() if self.add_lag_features else []
        self.history_numeric_columns_ = ["target_scaled", *numeric, *calendar_columns, *lag_columns]
        self.future_numeric_columns_ = [*numeric, *calendar_columns, *lag_columns]
        self.categorical_code_columns_ = [f"{col}__code" for col in categorical]
        self.categorical_cardinalities_ = [len(self.category_maps_[col]) for col in categorical]
        self.embedding_dims_ = [self._embedding_size(cardinality) for cardinality in self.categorical_cardinalities_]
        self.input_columns_ = [*self.history_numeric_columns_, *self.categorical_code_columns_]
        self.future_columns_ = [*self.future_numeric_columns_, *self.categorical_code_columns_]
        logger.info(
            "Prepared LSTM feature metadata numeric_covariates=%s categorical_covariates=%s "
            "categorical_cardinalities=%s embedding_dims=%s",
            self.numeric_covariates_,
            self.categorical_covariates_,
            self.categorical_cardinalities_,
            self.embedding_dims_,
        )

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
        data: TimeSeriesDataset,
        fit_scalers: bool,
        include_target: bool = True,
    ) -> pd.DataFrame:
        prepared = frame.copy()
        numeric = self.numeric_covariates_ or []
        categorical = self.categorical_covariates_ or []
        if self.add_calendar_features:
            prepared = self._add_calendar_features(prepared, data.timestamp_col)
        if include_target:
            if fit_scalers:
                self.target_scaler = StandardScaler()
                prepared["target_scaled"] = self.target_scaler.fit_transform(
                    prepared[[data.target_col]].astype(float)
                ).ravel()
            else:
                prepared["target_scaled"] = self.target_scaler.transform(
                    prepared[[data.target_col]].astype(float)
                ).ravel()
        if numeric:
            if fit_scalers:
                self.covariate_scaler = StandardScaler()
                prepared[numeric] = self.covariate_scaler.fit_transform(prepared[numeric].astype(float))
            else:
                prepared[numeric] = self.covariate_scaler.transform(prepared[numeric].astype(float))
        for col in categorical:
            mapping = self.category_maps_[col]
            prepared[f"{col}__code"] = prepared[col].astype(str).map(mapping).fillna(0).astype(int)
        if self.add_lag_features:
            prepared = self._add_lag_features(prepared, data)
        engineered_columns = self._engineered_feature_columns()
        if engineered_columns:
            if fit_scalers:
                self.engineered_feature_scaler = StandardScaler()
                prepared[engineered_columns] = self.engineered_feature_scaler.fit_transform(
                    prepared[engineered_columns].astype(float)
                )
            else:
                prepared[engineered_columns] = self.engineered_feature_scaler.transform(
                    prepared[engineered_columns].astype(float)
                )
        return prepared

    def _calendar_feature_columns(self) -> list[str]:
        return [
            "feature__calendar__dayofweek",
            "feature__calendar__is_weekend",
            "feature__calendar__month",
            "feature__calendar__day",
            "feature__calendar__hour",
            "feature__calendar__minute",
            "feature__time__sin_day",
            "feature__time__cos_day",
            "feature__time__sin_week",
            "feature__time__cos_week",
        ]

    def _lag_feature_columns(self) -> list[str]:
        return [f"feature__lag__{int(lag)}" for lag in self.lag_feature_steps]

    def _engineered_feature_columns(self) -> list[str]:
        columns: list[str] = []
        if self.add_calendar_features:
            columns.extend(self._calendar_feature_columns())
        if self.add_lag_features:
            columns.extend(self._lag_feature_columns())
        return columns

    def _add_calendar_features(self, frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        prepared = frame.copy()
        ts = pd.to_datetime(prepared[timestamp_col])
        minute_of_day = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy()
        prepared["feature__calendar__dayofweek"] = ts.dt.dayofweek.to_numpy(dtype=float)
        prepared["feature__calendar__is_weekend"] = (ts.dt.dayofweek >= 5).astype(float).to_numpy()
        prepared["feature__calendar__month"] = ts.dt.month.to_numpy(dtype=float)
        prepared["feature__calendar__day"] = ts.dt.day.to_numpy(dtype=float)
        prepared["feature__calendar__hour"] = ts.dt.hour.to_numpy(dtype=float)
        prepared["feature__calendar__minute"] = ts.dt.minute.to_numpy(dtype=float)
        prepared["feature__time__sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440)
        prepared["feature__time__cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440)
        prepared["feature__time__sin_week"] = np.sin(
            2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
        )
        prepared["feature__time__cos_week"] = np.cos(
            2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
        )
        return prepared

    def _add_lag_features(self, frame: pd.DataFrame, data: TimeSeriesDataset) -> pd.DataFrame:
        prepared = frame.copy()
        for lag in self.lag_feature_steps:
            column = f"feature__lag__{int(lag)}"
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).astype(float)
                continue
            if data.target_col not in prepared.columns:
                prepared[column] = 0.0
                continue
            prepared[column] = (
                prepared.groupby(data.item_id_col, sort=False)[data.target_col]
                .shift(int(lag))
                .fillna(0.0)
                .astype(float)
            )
        return prepared

    def _add_future_lag_features(
        self,
        future: pd.DataFrame,
        item_history: pd.DataFrame,
        history: TimeSeriesDataset,
    ) -> pd.DataFrame:
        prepared = future.copy()
        history_by_timestamp = (
            item_history[[history.timestamp_col, history.target_col]]
            .dropna(subset=[history.target_col])
            .assign(**{history.timestamp_col: lambda frame: pd.to_datetime(frame[history.timestamp_col])})
            .set_index(history.timestamp_col)[history.target_col]
        )
        freq_offset = pd.tseries.frequencies.to_offset(history.freq or self.freq)
        future_timestamps = pd.to_datetime(prepared[history.timestamp_col])
        for lag in self.lag_feature_steps:
            column = f"feature__lag__{int(lag)}"
            lagged_timestamps = future_timestamps - int(lag) * freq_offset
            prepared[column] = [
                float(history_by_timestamp.get(pd.Timestamp(timestamp), 0.0))
                for timestamp in lagged_timestamps
            ]
        return prepared

    def _build_training_arrays(
        self,
        prepared: pd.DataFrame,
        data: TimeSeriesDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_hist_num: list[np.ndarray] = []
        x_hist_cat: list[np.ndarray] = []
        x_future_num: list[np.ndarray] = []
        x_future_cat: list[np.ndarray] = []
        y: list[np.ndarray] = []
        for _, group in prepared.groupby(data.item_id_col, sort=False):
            group = group.sort_values(data.timestamp_col)
            max_start = len(group) - self.context_length - self.prediction_length + 1
            if max_start <= 0:
                continue
            starts = np.arange(max_start)
            if self.max_train_samples is not None and len(starts) > self.max_train_samples:
                starts = self._sample_starts(starts)
            history_numeric_values = group[self.history_numeric_columns_].to_numpy(dtype=float)
            future_numeric_values = group[self.future_numeric_columns_].to_numpy(dtype=float)
            categorical_values = group[self.categorical_code_columns_].to_numpy(dtype=int)
            target_values = group["target_scaled"].to_numpy(dtype=float)
            for start in starts:
                context_end = start + self.context_length
                horizon_end = context_end + self.prediction_length
                x_hist_num.append(history_numeric_values[start:context_end])
                x_hist_cat.append(categorical_values[start:context_end])
                x_future_num.append(future_numeric_values[context_end:horizon_end])
                x_future_cat.append(categorical_values[context_end:horizon_end])
                y.append(target_values[context_end:horizon_end])
        if not x_hist_num:
            raise ValueError(
                "Not enough rows to train LSTM: "
                f"context_length={self.context_length}, prediction_length={self.prediction_length}"
            )
        return (
            np.stack(x_hist_num),
            np.stack(x_hist_cat),
            np.stack(x_future_num),
            np.stack(x_future_cat),
            np.stack(y),
        )

    def _train_validation_slices(self, sample_count: int) -> tuple[slice, slice | None]:
        if (
            self.early_stopping_patience <= 0
            or self.validation_fraction <= 0
            or sample_count < 2
        ):
            return slice(None), None
        validation_count = int(round(sample_count * self.validation_fraction))
        validation_count = max(1, min(validation_count, sample_count - 1))
        train_count = sample_count - validation_count
        return slice(0, train_count), slice(train_count, sample_count)

    def _evaluate_loader_loss(self, torch: Any, loader: Any) -> float:
        losses = []
        self.model.eval()
        with torch.no_grad():
            for hist_num_batch, hist_cat_batch, future_num_batch, future_cat_batch, target_batch in loader:
                output = self.model(
                    hist_num_batch.to(self.device),
                    hist_cat_batch.to(self.device),
                    future_num_batch.to(self.device),
                    future_cat_batch.to(self.device),
                )
                weights, means, stds, point = output
                loss = self._loss(torch, target_batch.to(self.device), weights, means, stds, point)
                losses.append(float(loss.detach().cpu()))
        self.model.train()
        return float(np.mean(losses))

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
            "LSTM training progress epoch=%s/%s batch=%s/%s progress=%.1f%% batch_loss=%.6f avg_loss=%.6f learning_rate=%.8f",
            epoch,
            self.epochs,
            batch_index,
            total_batches,
            progress_pct,
            losses[-1],
            float(np.mean(losses)),
            learning_rate,
        )

    def _future_frame(
        self,
        item_history: pd.DataFrame,
        history: TimeSeriesDataset,
        item_id: str,
        freq: str,
        known_covariates: pd.DataFrame | None,
    ) -> pd.DataFrame:
        last_timestamp = pd.Timestamp(item_history[history.timestamp_col].max())
        future_index = pd.date_range(last_timestamp, periods=self.prediction_length + 1, freq=freq)[1:]
        future = pd.DataFrame(
            {
                history.item_id_col: item_id,
                history.timestamp_col: future_index,
            }
        )
        if known_covariates is not None and not known_covariates.empty:
            future = future.merge(
                known_covariates,
                on=[history.item_id_col, history.timestamp_col],
                how="left",
            )
        for col in self.known_covariates_names:
            if col not in future.columns:
                future[col] = 0.0
        if self.add_lag_features:
            future = self._add_future_lag_features(future, item_history, history)
        return future

    def _nll(self, torch: Any, target: Any, weights: Any, means: Any, stds: Any) -> Any:
        target = target.unsqueeze(-1).expand_as(means)
        log_component = -0.5 * ((target - means) / stds) ** 2 - torch.log(stds) - 0.5 * np.log(2 * np.pi)
        return -torch.logsumexp(torch.log(weights + 1e-12) + log_component, dim=-1).mean()

    def _loss(self, torch: Any, target: Any, weights: Any, means: Any, stds: Any, point: Any | None) -> Any:
        nll_loss = self._nll(torch, target, weights, means, stds)
        if point is None or self.point_loss_weight <= 0:
            return self.nll_loss_weight * nll_loss
        point_loss = torch.mean(torch.abs(target - point))
        return self.nll_loss_weight * nll_loss + self.point_loss_weight * point_loss

    def _sample_starts(self, starts: np.ndarray) -> np.ndarray:
        max_samples = int(self.max_train_samples)
        strategy = self.train_sample_strategy.lower()
        if strategy == "random":
            rng = np.random.default_rng(self.random_state)
            return np.sort(rng.choice(starts, size=max_samples, replace=False))
        if strategy == "first":
            return starts[:max_samples]
        if strategy == "last":
            return starts[-max_samples:]
        if strategy in {"uniform_time", "uniform"}:
            indices = np.linspace(0, len(starts) - 1, max_samples, dtype=int)
            return starts[indices]
        raise ValueError(
            "Unsupported train_sample_strategy "
            f"'{self.train_sample_strategy}'. Use random, first, last, or uniform_time."
        )

    def _embedding_size(self, cardinality: int) -> int:
        if self.embedding_dim is not None:
            return int(self.embedding_dim)
        return int(min(16, max(2, round(np.sqrt(max(cardinality, 1))))))

    def _inverse_target(self, values: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()

    def _inverse_target_stds(self, stds: np.ndarray) -> np.ndarray:
        scale = float(self.target_scaler.scale_[0])
        return np.maximum(np.asarray(stds, dtype=float) * scale, 1e-6)

    def _distribution_row(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        prediction: float,
    ) -> dict[str, object]:
        weights, means, stds = self._validate_mixture(weights, means, stds)
        lower, upper = self._mixture_quantiles(weights, means, stds)
        row: dict[str, object] = {
            "0.1": lower,
            "0.9": upper,
            "prediction_lower": lower,
            "prediction_upper": upper,
            "prediction_density": float(self._mixture_pdf(np.array([prediction]), weights, means, stds)[0]),
            "lstm_weights": json.dumps(weights.tolist()),
            "lstm_means": json.dumps(means.tolist()),
            "lstm_stds": json.dumps(stds.tolist()),
        }
        if self.distribution_grid_size > 1:
            grid = self._distribution_grid(prediction, means, stds)
            row["distribution_values"] = json.dumps(grid.tolist())
            row["distribution_probabilities"] = json.dumps(
                self._mixture_pdf(grid, weights, means, stds).tolist()
            )
        return row

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
        stds = np.maximum(stds.reshape(1, -1), 1e-6)
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
        stds = np.maximum(stds.reshape(1, -1), 1e-6)
        z = (values - means) / (stds * np.sqrt(2.0))
        component_cdf = 0.5 * (1.0 + np.vectorize(_erf)(z))
        return np.sum(component_cdf * weights.reshape(1, -1), axis=1)

    @staticmethod
    def _validate_mixture(
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        means = np.asarray(means, dtype=float).reshape(-1)
        stds = np.asarray(stds, dtype=float).reshape(-1)
        if not (len(weights) == len(means) == len(stds)):
            raise ValueError("LSTM mixture weights, means, and stds must have the same length")
        weights = np.clip(weights, 0.0, None)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            weights = np.full(len(weights), 1.0 / len(weights), dtype=float)
        else:
            weights = weights / total_weight
        return weights.astype(float), means.astype(float), np.maximum(stds, 1e-6).astype(float)

    def _metadata(self) -> dict[str, Any]:
        return {
            "prediction_length": self.prediction_length,
            "freq": self.freq,
            "known_covariates_names": self.known_covariates_names,
            "random_state": self.random_state,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler_enabled": self.lr_scheduler_enabled,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "min_learning_rate": self.min_learning_rate,
            "train_progress_log_interval": self.train_progress_log_interval,
            "max_train_samples": self.max_train_samples,
            "train_sample_strategy": self.train_sample_strategy,
            "device": self.device,
            "bidirectional": self.bidirectional,
            "embedding_dim": self.embedding_dim,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "n_components": self.n_components,
            "min_std": self.min_std,
            "use_point_head": self.use_point_head,
            "nll_loss_weight": self.nll_loss_weight,
            "point_loss_weight": self.point_loss_weight,
            "add_calendar_features": self.add_calendar_features,
            "add_lag_features": self.add_lag_features,
            "lag_feature_steps": self.lag_feature_steps,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "distribution_grid_size": self.distribution_grid_size,
            "distribution_std_width": self.distribution_std_width,
            "input_columns": self.input_columns_,
            "future_columns": self.future_columns_,
            "history_numeric_columns": self.history_numeric_columns_,
            "future_numeric_columns": self.future_numeric_columns_,
            "categorical_code_columns": self.categorical_code_columns_,
            "categorical_cardinalities": self.categorical_cardinalities_,
            "embedding_dims": self.embedding_dims_,
            "training_history": self.training_history_,
            "residual_std": self.residual_std_,
        }


def _erf(value: float) -> float:
    import math

    return math.erf(float(value))
