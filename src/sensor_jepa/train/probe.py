from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import get_device_name
from common.metrics import classification_metrics, flatten_metrics
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_milling import prepare_from_config
from sensor_jepa.data.windowing import stratified_label_fraction_indices
from sensor_jepa.train.pretrain import build_model_from_config


@torch.no_grad()
def extract_embeddings(model, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        outs.append(model.encode(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def load_pretrained_sensor(cfg: dict[str, Any], device: str):
    ckpt = torch.load(cfg["outputs"]["checkpoint"], map_location=device, weights_only=False)
    model = build_model_from_config(cfg, int(ckpt["input_channels"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def run_sensor_probe(cfg: dict[str, Any], probe_type: str = "linear", label_fraction: float | None = None) -> dict[str, Any]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_from_config(cfg, force=False)
    model = load_pretrained_sensor(cfg, device)
    z_train = extract_embeddings(model, bundle.x_train, device)
    z_test = extract_embeddings(model, bundle.x_test, device)
    y_train = bundle.y_train
    if label_fraction is None:
        label_fraction = float(cfg["training"].get("label_fraction", 1.0))
    idx = stratified_label_fraction_indices(y_train, label_fraction, seed=int(cfg.get("seed", 42)))
    z_fit, y_fit = z_train[idx], y_train[idx]
    start = time.time()
    if probe_type == "mlp":
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=int(cfg["training"].get("probe_max_iter", 500)), random_state=int(cfg.get("seed", 42))),
        )
    elif probe_type == "ridge":
        clf = make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced"))
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=int(cfg["training"].get("probe_max_iter", 500)), class_weight="balanced"),
        )
    clf.fit(z_fit, y_fit)
    pred = clf.predict(z_test)
    score = clf.predict_proba(z_test) if hasattr(clf, "predict_proba") else None
    metrics = flatten_metrics(classification_metrics(bundle.y_test, pred, score))
    metrics.update(
        {
            "dataset": "cnc_milling",
            "task": "wear_classification",
            "model_name": f"sensor_jepa_frozen_{probe_type}",
            "model_family": "jepa",
            "seed": cfg.get("seed", 42),
            "label_fraction": label_fraction,
            "encoder_mode": "frozen",
            "probe_type": probe_type,
            "frozen_encoder": True,
            "train_time_sec": time.time() - start,
        }
    )
    out_root = Path(cfg["outputs"]["root"])
    ensure_dir(out_root / "probe")
    out_csv = out_root / "probe" / f"sensor_probe_{probe_type}_{label_fraction}.csv"
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    write_markdown_report(
        out_root / "reports" / f"sensor_probe_{probe_type}_{label_fraction}.md",
        "Sensor-JEPA Frozen Probe",
        {
            "Metrics": markdown_table([metrics]),
            "Interpretation": "A frozen probe tests whether the encoder learned reusable representations.",
        },
    )
    return metrics

