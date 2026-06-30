from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .official_time_series_baselines import baseline_metadata_for_name


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def fit_tabular_models(seed: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
        )
    except Exception:
        pass
    return models


def run_tabular_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    xtr, xva, xte = _flatten(x_train), _flatten(x_val), _flatten(x_test)
    for name, model in fit_tabular_models(seed).items():
        start = time.time()
        try:
            model.fit(xtr, y_train)
            rows.append(
                {
                    "model_name": name,
                    "model_family": "tabular_or_feature",
                    "val_scores": predict_scores(model, xva),
                    "test_scores": predict_scores(model, xte),
                    "train_time_sec": time.time() - start,
                    "notes": "current_window_flat_features",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": name,
                    "model_family": "tabular_or_feature",
                    "val_scores": np.zeros(len(y_val)),
                    "test_scores": np.zeros(len(x_test)),
                    "train_time_sec": time.time() - start,
                    "notes": f"failed:{type(exc).__name__}",
                }
            )
    return rows


def run_matrix_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    model_prefix: str,
    model_family: str,
    notes: str,
    model_names: tuple[str, ...] = ("logistic_regression", "random_forest", "hist_gradient_boosting"),
) -> list[dict[str, Any]]:
    rows = []
    models = fit_tabular_models(seed)
    xtr = x_train.reshape(len(x_train), -1)
    xva = x_val.reshape(len(x_val), -1)
    xte = x_test.reshape(len(x_test), -1)
    for base_name in model_names:
        if base_name not in models:
            continue
        model = models[base_name]
        start = time.time()
        try:
            model.fit(xtr, y_train)
            rows.append(
                {
                    "model_name": f"{model_prefix}_{base_name}",
                    "model_family": model_family,
                    "val_scores": predict_scores(model, xva),
                    "test_scores": predict_scores(model, xte),
                    "train_time_sec": time.time() - start,
                    "notes": notes,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": f"{model_prefix}_{base_name}",
                    "model_family": model_family,
                    "val_scores": np.zeros(len(x_val)),
                    "test_scores": np.zeros(len(x_test)),
                    "train_time_sec": time.time() - start,
                    "notes": f"{notes};failed:{type(exc).__name__}",
                }
            )
    return rows


def run_official_rocket_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    # Optional path. The current environment has neither sktime nor aeon installed.
    # Inputs for these libraries are usually [N, C, T].
    rows: list[dict[str, Any]] = []
    xtr = np.transpose(x_train, (0, 2, 1))
    xva = np.transpose(x_val, (0, 2, 1))
    xte = np.transpose(x_test, (0, 2, 1))
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

        for name, transformer in [("minirocket_official", MiniRocket()), ("multirocket_official", MultiRocket())]:
            start = time.time()
            feats_train = transformer.fit_transform(xtr, y_train)
            feats_val = transformer.transform(xva)
            feats_test = transformer.transform(xte)
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
            clf.fit(feats_train, y_train)
            rows.append(
                {
                    "model_name": name,
                    "model_family": "rocket_official",
                    "val_scores": predict_scores(clf, feats_val),
                    "test_scores": predict_scores(clf, feats_test),
                    "train_time_sec": time.time() - start,
                    "notes": "aeon_official",
                    **baseline_metadata_for_name(name, "rocket_official", "aeon_official"),
                }
            )
        return rows
    except Exception:
        pass
    try:
        from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate

        for name, transformer in [("minirocket_official", MiniRocket()), ("minirocket_multivariate_official", MiniRocketMultivariate())]:
            start = time.time()
            feats_train = transformer.fit_transform(xtr, y_train)
            feats_val = transformer.transform(xva)
            feats_test = transformer.transform(xte)
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
            clf.fit(feats_train, y_train)
            rows.append(
                {
                    "model_name": name,
                    "model_family": "rocket_official",
                    "val_scores": predict_scores(clf, feats_val),
                    "test_scores": predict_scores(clf, feats_test),
                    "train_time_sec": time.time() - start,
                    "notes": "sktime_official",
                    **baseline_metadata_for_name(name, "rocket_official", "sktime_official"),
                }
            )
        return rows
    except Exception:
        return []


class TCNClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class GRUClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def train_torch_binary_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    device: str,
    seed: int,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    start = time.time()
    model.to(device)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()

    def scores(x: np.ndarray) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
                outs.append(F.softmax(model(xb), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(outs)

    return {
        "val_scores": scores(x_val),
        "test_scores": scores(x_test),
        "train_time_sec": time.time() - start,
    }


@dataclass
class RocketLiteTransformer:
    n_kernels: int = 512
    kernel_size: int = 7
    seed: int = 42

    def fit(self, x: np.ndarray) -> "RocketLiteTransformer":
        rng = np.random.default_rng(self.seed)
        channels = x.shape[-1]
        self.channels_ = rng.integers(0, channels, size=self.n_kernels)
        self.weights_ = rng.normal(0, 1, size=(self.n_kernels, self.kernel_size)).astype(np.float32)
        self.weights_ -= self.weights_.mean(axis=1, keepdims=True)
        self.bias_ = rng.normal(0, 1, size=self.n_kernels).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        feats = []
        for i in range(self.n_kernels):
            series = x[:, :, self.channels_[i]]
            conv = np.stack([np.convolve(s, self.weights_[i], mode="valid") for s in series], axis=0) + self.bias_[i]
            ppv = (conv > 0).mean(axis=1)
            maxv = conv.max(axis=1)
            feats.append(ppv)
            feats.append(maxv)
        return np.stack(feats, axis=1).astype(np.float32)


def run_rocket_lite_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    multi: bool = False,
) -> dict[str, Any]:
    start = time.time()
    n_kernels = 768 if multi else 512
    rocket = RocketLiteTransformer(n_kernels=n_kernels, kernel_size=7, seed=seed).fit(x_train)
    xtr = rocket.transform(x_train)
    xva = rocket.transform(x_val)
    xte = rocket.transform(x_test)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    clf.fit(xtr, y_train)
    return {
        "model_name": "multirocket_lite" if multi else "minirocket_lite",
        "model_family": "rocket_fallback",
        "val_scores": predict_scores(clf, xva),
        "test_scores": predict_scores(clf, xte),
        "train_time_sec": time.time() - start,
        "notes": "fallback_not_exact_sktime_or_aeon_missing",
        **baseline_metadata_for_name("multirocket_lite" if multi else "minirocket_lite", "rocket_fallback", "fallback_not_exact_sktime_or_aeon_missing"),
    }


class TemporalContrastiveEncoder(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def run_ts2vec_proxy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    device: str,
    seed: int,
    epochs: int = 8,
    batch_size: int = 64,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    start = time.time()
    model = TemporalContrastiveEncoder(x_train.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            noise1 = torch.randn_like(xb) * 0.03
            noise2 = torch.randn_like(xb) * 0.03
            z1 = F.normalize(model(xb + noise1), dim=-1)
            z2 = F.normalize(model(xb + noise2), dim=-1)
            logits = z1 @ z2.T / 0.2
            target = torch.arange(len(xb), device=device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def embed(x: np.ndarray) -> np.ndarray:
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                outs.append(model(torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)).cpu().numpy())
        return np.concatenate(outs)

    ztr, zva, zte = embed(x_train), embed(x_val), embed(x_test)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    clf.fit(ztr, y_train)
    return {
        "model_name": "ts2vec_proxy_temporal_contrastive",
        "model_family": "ssl_temporal_proxy",
        "val_scores": predict_scores(clf, zva),
        "test_scores": predict_scores(clf, zte),
        "train_time_sec": time.time() - start,
        "notes": "proxy_not_official_ts2vec",
        **baseline_metadata_for_name("ts2vec_proxy_temporal_contrastive", "ssl_temporal_proxy", "proxy_not_official_ts2vec"),
    }
