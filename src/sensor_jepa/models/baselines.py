from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.metrics import classification_metrics, flatten_metrics


class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 3, hidden_dim: int = 64):
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
    def __init__(self, input_channels: int, num_classes: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def train_torch_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    start = time.time()
    model.to(device)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()
    pred = logits.argmax(axis=1)
    row = flatten_metrics(classification_metrics(y_test, pred, logits))
    row["train_time_sec"] = time.time() - start
    return row


def run_sklearn_baselines(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> list[dict[str, Any]]:
    xtr = x_train.reshape(len(x_train), -1)
    xte = x_test.reshape(len(x_test), -1)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        "mlp_classifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)),
    }
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        start = time.time()
        model.fit(xtr, y_train)
        pred = model.predict(xte)
        score = model.predict_proba(xte) if hasattr(model, "predict_proba") else None
        row = flatten_metrics(classification_metrics(y_test, pred, score))
        row.update({"model_name": name, "model_family": "classic", "train_time_sec": time.time() - start})
        rows.append(row)
    return rows


def run_deep_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int = 6,
) -> list[dict[str, Any]]:
    rows = []
    for name, model in [
        ("cnn1d_supervised", CNN1DClassifier(x_train.shape[-1])),
        ("gru_supervised", GRUClassifier(x_train.shape[-1])),
    ]:
        row = train_torch_classifier(model, x_train, y_train, x_test, y_test, device=device, epochs=epochs)
        row.update({"model_name": name, "model_family": "deep_supervised"})
        rows.append(row)
    return rows

