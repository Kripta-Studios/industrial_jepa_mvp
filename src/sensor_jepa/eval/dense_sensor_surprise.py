from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import get_device_name
from common.forecast_metrics import forecast_metrics, threshold_from_validation
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.models.dense_sensor_jepa import DenseSensorJEPA
from sensor_jepa.models.strong_baselines import predict_scores


def surprise_scores_from_errors(errors: torch.Tensor, topk_ratio: float = 0.1) -> dict[str, torch.Tensor]:
    if errors.ndim != 2:
        raise ValueError(f"Expected [B, N] token errors, got {tuple(errors.shape)}")
    k = max(1, int(round(errors.shape[1] * topk_ratio)))
    alpha = 0.3
    ewma = errors[:, 0]
    for i in range(1, errors.shape[1]):
        ewma = alpha * errors[:, i] + (1.0 - alpha) * ewma
    row_mean = errors.mean(dim=1, keepdim=True)
    row_std = errors.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return {
        "avg_surprise": errors.mean(dim=1),
        "max_surprise": errors.max(dim=1).values,
        "topk_surprise": errors.topk(k, dim=1).values.mean(dim=1),
        "ewma_surprise": ewma,
        "surprise_slope": errors[:, -1] - errors[:, 0],
        "surprise_q90": torch.quantile(errors, 0.90, dim=1),
        "surprise_persistence": (errors > (row_mean + row_std)).float().mean(dim=1),
        "temporal_surprise_curve": errors,
    }


@torch.no_grad()
def score_dense_sensor_windows(
    model: DenseSensorJEPA,
    x: np.ndarray,
    device: str,
    batch_size: int = 256,
    topk_ratio: float = 0.1,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "avg_surprise": [],
        "max_surprise": [],
        "topk_surprise": [],
        "ewma_surprise": [],
        "surprise_slope": [],
        "surprise_q90": [],
        "surprise_persistence": [],
    }
    curves = []
    model.eval()
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        token_count = model.context_encoder(xb)["tokens"].shape[1]  # type: ignore[index]
        masks = model.make_masks(token_count, len(xb), xb.device, seed=0)
        out = model(xb, masks=masks)
        scores = surprise_scores_from_errors(out["token_prediction_error"], topk_ratio=topk_ratio)
        for key in rows:
            rows[key].append(scores[key].cpu().numpy())
        curves.append(scores["temporal_surprise_curve"].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()} | {"temporal_surprise_curve": np.concatenate(curves, axis=0)}


@torch.no_grad()
def dense_embedding_pool(model: DenseSensorJEPA, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    pooled = []
    model.eval()
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        tokens = model.encode_tokens(xb)
        weights = torch.softmax(tokens.norm(dim=-1), dim=1).unsqueeze(-1)
        attention_pool = (tokens * weights).sum(dim=1)
        pooled.append(
            torch.cat(
                [
                    tokens.mean(dim=1),
                    tokens.max(dim=1).values,
                    tokens.std(dim=1, unbiased=False),
                    attention_pool,
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(pooled, axis=0)


def build_dense_model_from_config(cfg: dict[str, Any], input_channels: int) -> DenseSensorJEPA:
    model_cfg = cfg.get("model", {})
    token_cfg = cfg.get("tokenization", {})
    mask_cfg = cfg.get("masking", {})
    loss_cfg = cfg.get("loss", {})
    return DenseSensorJEPA(
        input_channels=input_channels,
        encoder=model_cfg.get("encoder", "temporal_transformer"),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        depth=int(model_cfg.get("depth", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        temporal_patch_size=int(token_cfg.get("temporal_patch_size", 2)),
        temporal_patch_stride=int(token_cfg.get("temporal_patch_stride", 1)),
        tokenization_mode=token_cfg.get("mode", "multichannel_token"),
        target_mode=model_cfg.get("target_mode", "ema"),
        ema_momentum=float(model_cfg.get("ema_momentum", 0.996)),
        predictor=model_cfg.get("predictor", "transformer"),
        predictor_depth=int(model_cfg.get("predictor_depth", 2)),
        predictor_hidden_dim=int(model_cfg.get("predictor_hidden_dim", model_cfg.get("embedding_dim", 128) * 2)),
        temporal_mask_ratio=float(mask_cfg.get("temporal_mask_ratio", 0.4)),
        num_target_spans=int(mask_cfg.get("num_target_spans", 2)),
        min_span=int(mask_cfg.get("min_span", 1)),
        max_span=int(mask_cfg.get("max_span", 4)),
        target_loss_weight=float(loss_cfg.get("target_loss_weight", 1.0)),
        visible_loss_weight=float(loss_cfg.get("visible_loss_weight", 0.5)),
        future_loss_weight=float(loss_cfg.get("future_loss_weight", 0.0)),
        variance_weight=float(loss_cfg.get("variance_weight", 0.05)),
        covariance_weight=float(loss_cfg.get("covariance_weight", 0.0)),
        sigreg_weight=float(loss_cfg.get("sigreg_weight", 0.0)),
        sigreg_num_projections=int(loss_cfg.get("sigreg_num_projections", 128)),
        seed=int(cfg.get("seed", 42)),
    )


def _metadata_matrix(bundle: TransitionBundle, split: str) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    cols = [c for c in bundle.action_names if c in meta.columns]
    cols += [c for c in ["source_cycle"] if c in meta.columns]
    if not cols:
        return np.zeros((len(getattr(bundle, f"y_failure_{split}")), 1), dtype=np.float32)
    return meta[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _condition_groups(meta: pd.DataFrame) -> np.ndarray:
    if "ADOC" in meta.columns and "RDOC" in meta.columns:
        return (meta["ADOC"].astype(str) + "_rdoc_" + meta["RDOC"].astype(str)).to_numpy()
    if "ToolIndex" in meta.columns:
        return meta["ToolIndex"].astype(str).to_numpy()
    return np.array(["global"] * len(meta))


def _zscore_by_group(train_scores: np.ndarray, train_groups: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    global_mean = float(np.mean(train_scores))
    global_std = float(np.std(train_scores) + 1e-6)
    stats = {}
    for group in np.unique(train_groups):
        values = train_scores[train_groups == group]
        if len(values):
            stats[group] = (float(np.mean(values)), float(np.std(values) + 1e-6))
    out = np.empty_like(scores, dtype=np.float32)
    for i, (score, group) in enumerate(zip(scores, groups)):
        mean, std = stats.get(group, (global_mean, global_std))
        out[i] = (score - mean) / std
    return out


def _row_from_score(bundle: TransitionBundle, split_scores: dict[str, np.ndarray], score_name: str) -> dict[str, Any]:
    threshold = threshold_from_validation(bundle.y_failure_val, split_scores["val"])
    meta = bundle.test_meta
    row = forecast_metrics(
        bundle.y_failure_test,
        split_scores["test"],
        threshold=threshold,
        tool_ids=meta["ToolIndex"].to_numpy() if "ToolIndex" in meta else None,
        cycle_to_failure=meta["CycleToFailure"].to_numpy() if "CycleToFailure" in meta else None,
    )
    row.update({"model_name": f"dense_sensor_{score_name}", "score": score_name})
    return row


def _metadata_plus_feature_row(
    bundle: TransitionBundle,
    feature_train: np.ndarray,
    feature_val: np.ndarray,
    feature_test: np.ndarray,
    name: str,
    seed: int,
) -> dict[str, Any]:
    meta_train = _metadata_matrix(bundle, "train")
    meta_val = _metadata_matrix(bundle, "val")
    meta_test = _metadata_matrix(bundle, "test")
    x_train = np.concatenate([meta_train, np.atleast_2d(feature_train).reshape(len(feature_train), -1)], axis=1)
    x_val = np.concatenate([meta_val, np.atleast_2d(feature_val).reshape(len(feature_val), -1)], axis=1)
    x_test = np.concatenate([meta_test, np.atleast_2d(feature_test).reshape(len(feature_test), -1)], axis=1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    clf.fit(x_train, bundle.y_failure_train)
    val_scores = predict_scores(clf, x_val)
    test_scores = predict_scores(clf, x_test)
    threshold = threshold_from_validation(bundle.y_failure_val, val_scores)
    row = forecast_metrics(bundle.y_failure_test, test_scores, threshold)
    row.update({"model_name": name, "score": name.replace("metadata_plus_", "")})
    return row


def _feature_probe_row(
    bundle: TransitionBundle,
    feature_train: np.ndarray,
    feature_val: np.ndarray,
    feature_test: np.ndarray,
    name: str,
    seed: int,
    metadata: bool = False,
    mlp: bool = False,
) -> dict[str, Any]:
    if metadata:
        feature_train = np.concatenate([_metadata_matrix(bundle, "train"), feature_train], axis=1)
        feature_val = np.concatenate([_metadata_matrix(bundle, "val"), feature_val], axis=1)
        feature_test = np.concatenate([_metadata_matrix(bundle, "test"), feature_test], axis=1)
    if mlp:
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, random_state=seed, early_stopping=True),
        )
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    clf.fit(feature_train, bundle.y_failure_train)
    val_scores = predict_scores(clf, feature_val)
    test_scores = predict_scores(clf, feature_test)
    threshold = threshold_from_validation(bundle.y_failure_val, val_scores)
    row = forecast_metrics(bundle.y_failure_test, test_scores, threshold)
    row.update({"model_name": name, "score": "dense_embedding_pool", "probe": "mlp" if mlp else "logistic"})
    return row


def evaluate_dense_sensor_surprise(cfg: dict[str, Any]) -> dict[str, Path]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_transition_from_config(cfg)
    model = build_dense_model_from_config(cfg, bundle.input_channels).to(device)
    out_dir = ensure_dir(cfg.get("eval", {}).get("output_dir", cfg.get("outputs", {}).get("root", "outputs/sensor_jepa/dense_sensor_jepa_cnc")))
    ckpt_path = Path(cfg.get("eval", {}).get("checkpoint", out_dir / "checkpoints" / "latest.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DenseSensorJEPA checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    train_scores = score_dense_sensor_windows(model, bundle.x_train, device, batch_size=batch_size)
    val_scores = score_dense_sensor_windows(model, bundle.x_val, device, batch_size=batch_size)
    test_scores = score_dense_sensor_windows(model, bundle.x_test, device, batch_size=batch_size)
    rows = []
    base_score_names = [
        "avg_surprise",
        "max_surprise",
        "topk_surprise",
        "ewma_surprise",
        "surprise_slope",
        "surprise_q90",
        "surprise_persistence",
    ]
    for name in base_score_names:
        rows.append(_row_from_score(bundle, {"val": val_scores[name], "test": test_scores[name]}, name))
        rows.append(_row_from_score(bundle, {"val": -val_scores[name], "test": -test_scores[name]}, f"negative_{name}"))

    train_tool = bundle.train_meta["ToolIndex"].astype(str).to_numpy() if "ToolIndex" in bundle.train_meta else np.array(["global"] * len(bundle.y_failure_train))
    val_tool = bundle.val_meta["ToolIndex"].astype(str).to_numpy() if "ToolIndex" in bundle.val_meta else np.array(["global"] * len(bundle.y_failure_val))
    test_tool = bundle.test_meta["ToolIndex"].astype(str).to_numpy() if "ToolIndex" in bundle.test_meta else np.array(["global"] * len(bundle.y_failure_test))
    train_condition = _condition_groups(bundle.train_meta)
    val_condition = _condition_groups(bundle.val_meta)
    test_condition = _condition_groups(bundle.test_meta)
    for name in base_score_names:
        z_val_tool = _zscore_by_group(train_scores[name], train_tool, val_scores[name], val_tool)
        z_test_tool = _zscore_by_group(train_scores[name], train_tool, test_scores[name], test_tool)
        rows.append(_row_from_score(bundle, {"val": z_val_tool, "test": z_test_tool}, f"zscore_by_tool_{name}"))
        z_val_cond = _zscore_by_group(train_scores[name], train_condition, val_scores[name], val_condition)
        z_test_cond = _zscore_by_group(train_scores[name], train_condition, test_scores[name], test_condition)
        rows.append(_row_from_score(bundle, {"val": z_val_cond, "test": z_test_cond}, f"zscore_by_condition_{name}"))

    meta_train = _metadata_matrix(bundle, "train")
    meta_val = _metadata_matrix(bundle, "val")
    meta_test = _metadata_matrix(bundle, "test")
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(cfg.get("seed", 42))))
    clf.fit(meta_train, bundle.y_failure_train)
    meta_val_scores = predict_scores(clf, meta_val)
    meta_test_scores = predict_scores(clf, meta_test)
    meta_th = threshold_from_validation(bundle.y_failure_val, meta_val_scores)
    meta_row = forecast_metrics(bundle.y_failure_test, meta_test_scores, meta_th)
    meta_row.update({"model_name": "metadata_only", "score": "metadata"})
    rows.append(meta_row)
    seed = int(cfg.get("seed", 42))
    for feature_name in ["topk_surprise", "ewma_surprise", "surprise_slope", "surprise_persistence"]:
        rows.append(
            _metadata_plus_feature_row(
                bundle,
                train_scores[feature_name],
                val_scores[feature_name],
                test_scores[feature_name],
                f"metadata_plus_{feature_name}",
                seed,
            )
        )
        rows.append(
            _metadata_plus_feature_row(
                bundle,
                -train_scores[feature_name],
                -val_scores[feature_name],
                -test_scores[feature_name],
                f"metadata_plus_negative_{feature_name}",
                seed,
            )
        )
    train_pool = dense_embedding_pool(model, bundle.x_train, device, batch_size=batch_size)
    val_pool = dense_embedding_pool(model, bundle.x_val, device, batch_size=batch_size)
    test_pool = dense_embedding_pool(model, bundle.x_test, device, batch_size=batch_size)
    rows.append(_feature_probe_row(bundle, train_pool, val_pool, test_pool, "dense_embedding_pool_logistic", seed, metadata=False, mlp=False))
    rows.append(_feature_probe_row(bundle, train_pool, val_pool, test_pool, "dense_embedding_pool_mlp", seed, metadata=False, mlp=True))
    rows.append(_feature_probe_row(bundle, train_pool, val_pool, test_pool, "metadata_plus_dense_embedding_pool", seed, metadata=True, mlp=False))
    rows.append(_feature_probe_row(bundle, train_pool, val_pool, test_pool, "metadata_plus_dense_embedding_pool_mlp", seed, metadata=True, mlp=True))

    surprise_path = out_dir / "surprise_scores.csv"
    curves = test_scores.pop("temporal_surprise_curve")
    surprise_df = pd.DataFrame({k: v for k, v in test_scores.items()})
    surprise_df["label"] = bundle.y_failure_test
    meta = bundle.test_meta
    if "ToolIndex" in meta:
        surprise_df["tool_id"] = meta["ToolIndex"].to_numpy()
    surprise_df.to_csv(surprise_path, index=False)
    np.savez_compressed(out_dir / "temporal_surprise_curves.npz", temporal_surprise_curve=curves)

    results_path = out_dir / "incremental_results.csv"
    results_df = pd.DataFrame(rows).sort_values("AUPRC", ascending=False, na_position="last")
    results_df.to_csv(results_path, index=False)
    report_path = out_dir / "report.md"
    sanity_report_path = out_dir / "surprise_sanity_report.md"
    diagnostic_report_path = out_dir / "dense_diagnostic_report.md"
    topk = results_df[results_df["score"].eq("topk_surprise")]
    neg_topk = results_df[results_df["score"].eq("negative_topk_surprise")]
    write_markdown_report(
        report_path,
        "DenseSensorJEPA Surprise Report",
        {
            "Status": "Dense surprise scores were evaluated against metadata-only. Success still requires positive delta over metadata/cycle.",
            "TopK Surprise": f"AUPRC={float(topk.iloc[0]['AUPRC']) if len(topk) else None}, AUROC={float(topk.iloc[0]['AUROC']) if len(topk) else None}",
            "Outputs": f"`{surprise_path}`\n\n`{results_path}`",
        },
    )
    write_markdown_report(
        sanity_report_path,
        "DenseSensorJEPA Surprise Sanity Report",
        {
            "Direct Answer": (
                "This sanity check evaluates raw, inverted, group-normalized and metadata-combined DenseSensor scores. "
                "A failed direct surprise score is not treated as final evidence against dense embeddings."
            ),
            "TopK Surprise": f"AUPRC={float(topk.iloc[0]['AUPRC']) if len(topk) else None}, AUROC={float(topk.iloc[0]['AUROC']) if len(topk) else None}",
            "Negative TopK Surprise": f"AUPRC={float(neg_topk.iloc[0]['AUPRC']) if len(neg_topk) else None}, AUROC={float(neg_topk.iloc[0]['AUROC']) if len(neg_topk) else None}",
            "Top Rows": markdown_table(results_df.head(20).to_dict("records")),
        },
    )
    dense_rows = results_df[results_df["model_name"].astype(str).str.contains("dense_embedding_pool", regex=False)]
    write_markdown_report(
        diagnostic_report_path,
        "DenseSensorJEPA Diagnostic Report",
        {
            "Status": "Diagnostic only. DenseSensorJEPA should stay paused unless these features improve over current_z or sensor_raw in no-cycle/hard protocols.",
            "Pooling": "mean, max, std, and norm-weighted attention pooling over dense tokens.",
            "Probe Families": "LogisticRegression and one-hidden-layer MLP with StandardScaler.",
            "Dense Pool Rows": markdown_table(dense_rows.to_dict("records")) if len(dense_rows) else "No dense pool rows.",
            "Interpretation": "If metadata_plus_dense_pool is near metadata_only and dense_pool_only is weak, DenseSensor has no current evidence.",
        },
    )
    return {
        "surprise_scores": surprise_path,
        "incremental_results": results_path,
        "report": report_path,
        "sanity_report": sanity_report_path,
        "diagnostic_report": diagnostic_report_path,
    }
