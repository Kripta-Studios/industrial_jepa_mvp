from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from common.config import get_device_name, load_config
from common.forecast_metrics import forecast_metrics, threshold_from_validation
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.eval.dense_sensor_surprise import build_dense_model_from_config
from sensor_jepa.models.token_world_model import TokenWorldModel, token_surprise


REAL_ACTION_HINTS = {"setpoint", "command", "control", "spindle", "speed_cmd", "feed_cmd", "pressure_cmd"}


@torch.no_grad()
def _encode(model, x: np.ndarray, device: str, batch_size: int) -> torch.Tensor:
    outs = []
    model.eval()
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        outs.append(model.encode_tokens(xb).detach().cpu())
    return torch.cat(outs, dim=0)


def _subset(bundle: TransitionBundle, max_samples: int | None) -> TransitionBundle:
    if not max_samples:
        return bundle

    def take(arr):
        return arr[:max_samples]

    bundle.x_train = take(bundle.x_train)
    bundle.a_train = take(bundle.a_train)
    bundle.x_next_train = take(bundle.x_next_train)
    bundle.y_failure_train = take(bundle.y_failure_train)
    bundle.train_meta = bundle.train_meta.iloc[:max_samples].reset_index(drop=True)
    return bundle


def _scores_from_surprise(scores: torch.Tensor) -> dict[str, np.ndarray]:
    k = max(1, int(round(scores.shape[1] * 0.1)))
    alpha = 0.3
    ewma = scores[:, 0]
    for i in range(1, scores.shape[1]):
        ewma = alpha * scores[:, i] + (1.0 - alpha) * ewma
    return {
        "surprise_avg": scores.mean(dim=1).numpy(),
        "surprise_max": scores.max(dim=1).values.numpy(),
        "surprise_topk": scores.topk(k, dim=1).values.mean(dim=1).numpy(),
        "surprise_ewma": ewma.numpy(),
    }


def _metadata_matrix(bundle: TransitionBundle, split: str) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    cols = [c for c in bundle.action_names if c in meta.columns]
    cols += [c for c in ["source_cycle"] if c in meta.columns]
    if not cols:
        return np.zeros((len(getattr(bundle, f"y_failure_{split}")), 1), dtype=np.float32)
    return meta[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _predict_scores(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    scores = clf.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def _metadata_plus_feature_row(
    bundle: TransitionBundle,
    feature_train: np.ndarray,
    feature_val: np.ndarray,
    feature_test: np.ndarray,
    name: str,
    seed: int,
) -> dict[str, Any]:
    x_train = np.concatenate([_metadata_matrix(bundle, "train"), feature_train.reshape(len(feature_train), -1)], axis=1)
    x_val = np.concatenate([_metadata_matrix(bundle, "val"), feature_val.reshape(len(feature_val), -1)], axis=1)
    x_test = np.concatenate([_metadata_matrix(bundle, "test"), feature_test.reshape(len(feature_test), -1)], axis=1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    clf.fit(x_train, bundle.y_failure_train)
    val_scores = _predict_scores(clf, x_val)
    test_scores = _predict_scores(clf, x_test)
    threshold = threshold_from_validation(bundle.y_failure_val, val_scores)
    row = forecast_metrics(
        bundle.y_failure_test,
        test_scores,
        threshold=threshold,
        tool_ids=bundle.test_meta["ToolIndex"].to_numpy() if "ToolIndex" in bundle.test_meta else None,
        cycle_to_failure=bundle.test_meta["CycleToFailure"].to_numpy() if "CycleToFailure" in bundle.test_meta else None,
    )
    row.update({"protocol": "metadata_plus", "model_name": name, "score": name.replace("metadata_plus_", "")})
    return row


def _train_protocol(
    protocol: str,
    z_train: torch.Tensor,
    z_next_train: torch.Tensor,
    a_train: np.ndarray,
    z_val: torch.Tensor,
    z_next_val: torch.Tensor,
    a_val: np.ndarray,
    z_test: torch.Tensor,
    z_next_test: torch.Tensor,
    a_test: np.ndarray,
    bundle: TransitionBundle,
    device: str,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    action_dim = 0 if protocol == "no_action" else int(a_train.shape[-1])
    model = TokenWorldModel(embedding_dim=z_train.shape[-1], action_dim=action_dim, hidden_dim=max(128, z_train.shape[-1] * 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if action_dim:
        train_ds = TensorDataset(z_train, z_next_train, torch.tensor(a_train, dtype=torch.float32))
    else:
        train_ds = TensorDataset(z_train, z_next_train, torch.empty(len(z_train), 0))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for z, zn, a in loader:
            z, zn, a = z.to(device), zn.to(device), a.to(device)
            pred = model(z, a if action_dim else None, horizon=horizon)
            loss = F.mse_loss(pred, zn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(z)
            n += len(z)
        history.append({"protocol": protocol, "epoch": epoch, "prediction_loss": total / max(n, 1)})

    @torch.no_grad()
    def predict_scores(z, zn, a_np):
        model.eval()
        out = []
        for i in range(0, len(z), batch_size):
            zb = z[i : i + batch_size].to(device)
            znb = zn[i : i + batch_size].to(device)
            if action_dim:
                ab = torch.tensor(a_np[i : i + batch_size], dtype=torch.float32, device=device)
            else:
                ab = None
            out.append(token_surprise(model(zb, ab, horizon=horizon), znb).cpu())
        return _scores_from_surprise(torch.cat(out, dim=0))

    val_scores = predict_scores(z_val, z_next_val, a_val)
    test_scores = predict_scores(z_test, z_next_test, a_test)
    train_scores = predict_scores(z_train, z_next_train, a_train)
    rows = []
    for score_name in ["surprise_avg", "surprise_max", "surprise_topk", "surprise_ewma"]:
        for label, sign in [(score_name, 1.0), (f"negative_{score_name}", -1.0)]:
            val_score = sign * val_scores[score_name]
            test_score = sign * test_scores[score_name]
            threshold = threshold_from_validation(bundle.y_failure_val, val_score)
            row = forecast_metrics(
                bundle.y_failure_test,
                test_score,
                threshold=threshold,
                tool_ids=bundle.test_meta["ToolIndex"].to_numpy() if "ToolIndex" in bundle.test_meta else None,
                cycle_to_failure=bundle.test_meta["CycleToFailure"].to_numpy() if "CycleToFailure" in bundle.test_meta else None,
            )
            row.update({"protocol": protocol, "model_name": f"token_world_model_{protocol}_{label}", "score": label})
            rows.append(row)
            rows.append(
                _metadata_plus_feature_row(
                    bundle,
                    sign * train_scores[score_name],
                    val_score,
                    test_score,
                    f"metadata_plus_token_world_model_{protocol}_{label}",
                    seed,
                )
            )
    return history, {"protocol": protocol, "rows": rows, "model_state": model.state_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-level DenseSensor world model variants.")
    parser.add_argument("--config", default="configs/sensor_jepa/dense_sensor_cnc.yaml")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device_name(cfg.get("device", "auto"))
    out_dir = ensure_dir(args.out_root or cfg.get("token_world_model", {}).get("output_dir", "outputs/sensor_jepa/token_world_model"))
    bundle = _subset(prepare_transition_from_config(cfg), args.max_samples)
    dense = build_dense_model_from_config(cfg, bundle.input_channels).to(device)
    ckpt_path = Path(cfg.get("eval", {}).get("checkpoint", "outputs/sensor_jepa/dense_sensor_jepa_cnc/checkpoints/latest.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DenseSensorJEPA checkpoint not found: {ckpt_path}. Run scripts/25_pretrain_dense_sensor_jepa.py first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dense.load_state_dict(ckpt["model_state"], strict=True)
    dense.eval()
    for p in dense.parameters():
        p.requires_grad_(False)

    batch_size = args.batch_size
    z_train = _encode(dense, bundle.x_train, device, batch_size)
    z_next_train = _encode(dense, bundle.x_next_train, device, batch_size)
    z_val = _encode(dense, bundle.x_val, device, batch_size)
    z_next_val = _encode(dense, bundle.x_next_val, device, batch_size)
    z_test = _encode(dense, bundle.x_test, device, batch_size)
    z_next_test = _encode(dense, bundle.x_next_test, device, batch_size)

    histories: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    horizon = int(cfg.get("world_model", {}).get("forecast_horizon", 1))
    for protocol in ["no_action", "context"]:
        history, result = _train_protocol(
            protocol,
            z_train,
            z_next_train,
            bundle.a_train,
            z_val,
            z_next_val,
            bundle.a_val,
            z_test,
            z_next_test,
            bundle.a_test,
            bundle,
            device,
            horizon,
            args.epochs,
            batch_size,
            args.lr,
            int(cfg.get("seed", 42)),
        )
        histories.extend(history)
        result_rows.extend(result["rows"])
        states[protocol] = result["model_state"]

    action_names = set(name.lower() for name in bundle.action_names)
    has_real_actions = bool(action_names & REAL_ACTION_HINTS)
    pending = []
    if not has_real_actions:
        pending.append(
            {
                "protocol": "real_action",
                "model_name": "token_world_model_real_action",
                "status": "pending",
                "notes": "No real temporal control/setpoint action columns detected; CNC columns are treated as context/process variables.",
            }
        )
    results_df = pd.DataFrame(result_rows + pending)
    history_df = pd.DataFrame(histories)
    results_df.to_csv(Path(out_dir) / "results.csv", index=False)
    history_df.to_csv(Path(out_dir) / "train_log.csv", index=False)
    torch.save({"states": states, "cfg": cfg, "action_names": bundle.action_names}, Path(out_dir) / "token_world_model.pt")
    write_markdown_report(
        Path(out_dir) / "report.md",
        "Token Sensor World Model Report",
        {
            "Status": "Token-level world model trained on frozen DenseSensorJEPA tokens.",
            "Protocols": "no_action and context executed. real_action is only valid with real temporal setpoints/control commands.",
            "Action Columns": ", ".join(bundle.action_names),
            "Results": markdown_table(results_df.to_dict("records")),
            "Interpretation": "Compare these rows to metadata/cycle and engineered sensor baselines before making any product claim.",
        },
    )
    print(f"results: {Path(out_dir) / 'results.csv'}")
    print(f"report: {Path(out_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
