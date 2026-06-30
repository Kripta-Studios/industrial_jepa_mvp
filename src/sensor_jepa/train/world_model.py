from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import get_device_name
from common.metrics import anomaly_metrics, flatten_metrics
from common.paths import ensure_dir
from common.plots import plot_history
from common.reports import markdown_table, write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.models.sensor_world_model import ActionConditionedSensorWorldModel


def build_world_model_from_config(cfg: dict[str, Any], input_channels: int, action_dim: int) -> ActionConditionedSensorWorldModel:
    m = cfg["model"]
    wm = cfg.get("world_model", {})
    return ActionConditionedSensorWorldModel(
        input_channels=input_channels,
        action_dim=action_dim,
        encoder=m.get("encoder", "conv1d"),
        embedding_dim=int(m.get("embedding_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        predictor_hidden_dim=int(m.get("predictor_hidden_dim", 256)),
        sigreg_weight=float(wm.get("sigreg_weight", m.get("sigreg_weight", 0.05))),
    )


def _make_loader(bundle: TransitionBundle, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    x = getattr(bundle, f"x_{split}")
    a = getattr(bundle, f"a_{split}")
    x_next = getattr(bundle, f"x_next_{split}")
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(x_next, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def pretrain_sensor_world_model(cfg: dict[str, Any]) -> tuple[Path, list[dict[str, float]], TransitionBundle]:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_transition_from_config(cfg)
    model = build_world_model_from_config(cfg, bundle.input_channels, bundle.action_dim).to(device)
    train_cfg = cfg["training"]
    wm_cfg = cfg.get("world_model", {})
    loader = _make_loader(bundle, "train", int(train_cfg.get("batch_size", 64)), shuffle=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(wm_cfg.get("learning_rate", train_cfg.get("learning_rate", 1e-3))),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(wm_cfg.get("epochs", train_cfg.get("pretrain_epochs", 8)))
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        agg = {"loss": 0.0, "pred_loss": 0.0, "sigreg": 0.0, "embedding_std": 0.0}
        n = 0
        for xb, ab, xnb in loader:
            xb, ab, xnb = xb.to(device), ab.to(device), xnb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb, ab, xnb)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = len(xb)
            n += bs
            for k in agg:
                agg[k] += float(out[k].detach().cpu()) * bs
        history.append({"epoch": epoch, **{k: v / max(n, 1) for k, v in agg.items()}})

    out_root = Path(cfg["outputs"]["root"])
    ckpt_path = out_root / "checkpoints" / "sensor_action_world_model.pt"
    ensure_dir(ckpt_path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "input_channels": bundle.input_channels,
            "action_dim": bundle.action_dim,
            "feature_names": bundle.feature_names,
            "action_names": bundle.action_names,
            "history": history,
        },
        ckpt_path,
    )
    pd.DataFrame(history).to_csv(out_root / "world_model_history.csv", index=False)
    plot_history(history, out_root / "plots" / "sensor_action_world_model_loss.png", y_key="loss")
    write_markdown_report(
        out_root / "reports" / "sensor_action_world_model_pretrain.md",
        "Sensor Action-Conditioned World Model",
        {
            "LeWorldModel Mapping": (
                "State is a CNC sensor-feature window. Action/context is process metadata "
                f"`{', '.join(bundle.action_names)}`. The predictor learns `z_t, a_t -> z_t+h`."
            ),
            "Training": f"Train transitions: `{len(bundle.x_train)}`\n\nFinal loss: `{history[-1]['loss']:.6f}`\n\nElapsed seconds: `{time.time() - start:.2f}`",
        },
    )
    return ckpt_path, history, bundle


def load_sensor_world_model(cfg: dict[str, Any], device: str):
    ckpt_path = Path(cfg["outputs"]["root"]) / "checkpoints" / "sensor_action_world_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_world_model_from_config(cfg, int(ckpt["input_channels"]), int(ckpt["action_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _prediction_scores(model, x: np.ndarray, a: np.ndarray, x_next: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    scores = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        ab = torch.tensor(a[i : i + batch_size], dtype=torch.float32, device=device)
        xnb = torch.tensor(x_next[i : i + batch_size], dtype=torch.float32, device=device)
        out = model(xb, ab, xnb)
        scores.append(out["per_sample_error"].cpu().numpy())
    return np.concatenate(scores, axis=0)


@torch.no_grad()
def _predicted_future_embeddings(model, x: np.ndarray, a: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    embeddings = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        ab = torch.tensor(a[i : i + batch_size], dtype=torch.float32, device=device)
        embeddings.append(model.predict_next_embedding(xb, ab).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def evaluate_sensor_world_model(cfg: dict[str, Any], bundle: TransitionBundle | None = None) -> dict[str, Any]:
    device = get_device_name(cfg.get("device", "auto"))
    if bundle is None:
        bundle = prepare_transition_from_config(cfg)
    model = load_sensor_world_model(cfg, device)
    train_scores = _prediction_scores(model, bundle.x_train, bundle.a_train, bundle.x_next_train, device)
    val_scores = _prediction_scores(model, bundle.x_val, bundle.a_val, bundle.x_next_val, device)
    test_scores = _prediction_scores(model, bundle.x_test, bundle.a_test, bundle.x_next_test, device)
    try:
        val_raw_auc = float(roc_auc_score(bundle.y_failure_val, val_scores))
    except Exception:
        val_raw_auc = 0.5
    orientation = "raw_error" if val_raw_auc >= 0.5 else "inverse_error"
    train_alert_scores = train_scores if orientation == "raw_error" else -train_scores
    test_alert_scores = test_scores if orientation == "raw_error" else -test_scores
    healthy_train_scores = train_alert_scores[bundle.y_failure_train == 0]
    threshold = float(np.quantile(healthy_train_scores if len(healthy_train_scores) else train_alert_scores, 0.95))
    metrics = flatten_metrics(anomaly_metrics(bundle.y_failure_test, test_alert_scores, threshold=threshold, prefix="failure_soon_"))
    try:
        metrics["failure_soon_raw_error_AUROC"] = float(roc_auc_score(bundle.y_failure_test, test_scores))
    except Exception:
        metrics["failure_soon_raw_error_AUROC"] = None
    metrics["validation_raw_error_AUROC"] = val_raw_auc
    metrics["score_orientation"] = orientation

    z_train_pred = _predicted_future_embeddings(model, bundle.x_train, bundle.a_train, device)
    z_test_pred = _predicted_future_embeddings(model, bundle.x_test, bundle.a_test, device)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(cfg.get("seed", 42))),
    )
    probe.fit(z_train_pred, bundle.y_failure_train)
    probe_scores = probe.predict_proba(z_test_pred)[:, 1]
    probe_metrics = flatten_metrics(anomaly_metrics(bundle.y_failure_test, probe_scores, threshold=0.5, prefix="forecast_probe_"))
    metrics.update(probe_metrics)
    wm_cfg = cfg.get("world_model", {})
    metrics.update(
        {
            "dataset": "cnc_milling",
            "task": "failure_soon_prediction",
            "model_name": "sensor_action_world_model",
            "model_family": "jepa_world_model",
            "seed": cfg.get("seed", 42),
            "forecast_horizon": wm_cfg.get("forecast_horizon", 1),
            "failure_horizon_cycles": wm_cfg.get("failure_horizon_cycles", 10),
            "action_columns": ",".join(bundle.action_names),
            "test_failure_rate": float(np.mean(bundle.y_failure_test)) if len(bundle.y_failure_test) else 0.0,
        }
    )
    out_root = Path(cfg["outputs"]["root"])
    ensure_dir(out_root / "world_model")
    pd.DataFrame([metrics]).to_csv(out_root / "world_model" / "world_model_results.csv", index=False)
    score_df = bundle.test_meta.copy()
    score_df["failure_soon_label"] = bundle.y_failure_test
    score_df["latent_prediction_error"] = test_scores
    score_df["oriented_alert_score"] = test_alert_scores
    score_df["forecast_probe_failure_probability"] = probe_scores
    score_df["alert"] = (test_alert_scores >= threshold).astype(int)
    score_df["forecast_probe_alert"] = (probe_scores >= 0.5).astype(int)
    score_df.to_csv(out_root / "world_model" / "world_model_test_scores.csv", index=False)
    write_markdown_report(
        out_root / "reports" / "sensor_action_world_model_eval.md",
        "Sensor Action World Model Evaluation",
        {
            "Metrics": markdown_table([metrics]),
            "Interpretation": (
                "This is the LeWorldModel-style part: `z_t, a_t -> z_t+h`. The alert score is oriented on "
                "validation data. If `score_orientation` is `inverse_error`, the learned dynamics are more predictable "
                "near failure in this split, so low raw error rather than high surprise is associated with failure-soon. "
                "The `forecast_probe_*` metrics use the predicted future embedding with a lightweight supervised probe, "
                "which is the recommended failure-forecasting readout."
            ),
        },
    )
    return metrics


def run_sensor_world_model(cfg: dict[str, Any]) -> dict[str, Any]:
    pretrain_sensor_world_model(cfg)
    return evaluate_sensor_world_model(cfg)
