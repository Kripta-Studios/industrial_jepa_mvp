# Industrial JEPA MVP: Project Overview

This document provides a comprehensive overview of the `industrial_jepa_mvp` project. It describes the initial goals, datasets, architecture, implemented modules, and lists what is functional, what is experimental or underperforming, and what remains pending.

## 1. Initial Project Goals
The project was launched to explore Joint Embedding Predictive Architectures (JEPA) and latent predictive models for industrial applications. It operates in two main lines:
1. **Sensor-JEPA / Predictive Maintenance**: Exploring latent predictive models for CNC tool wear and failure forecasting. The objective was to predict future latent representations from current representations and process actions (inspired by world models like LeWorldModel), using the prediction error (surprise) or learned features to forecast imminent failures.
2. **Visual-JEPA / Industrial Visual Anomaly Detection**: Exploring self-supervised representations on visual datasets (MVTec AD, VisA, KolektorSDD) to spot defects without manual labeling, matching or outperforming standard models.

## 2. Datasets Used
* **CNC Milling Dataset (Fully Implemented)**: High-resolution signals (current, vibration, acoustic emission) across multiple cutting cycles, tools, and materials.
* **MVTec AD (Bottle Category Fully Implemented)**: Standard benchmark for visual industrial anomalies.
* **CWRU Bearing / Paderborn Bearing (Partially Implemented/Scaffolded)**: Sensor datasets for bearing defect detection, currently integrated with manifest checks but pending full training execution.
* **VisA / KolektorSDD (Partially Implemented/Scaffolded)**: Data loader scaffolds exist for training but full benchmarks are pending.

## 3. Repository Structure
* [src/sensor_jepa/](file:///c:/Users/Álvaro%20Schwiedop/Desktop/KriptaStudios/industrial_jepa_mvp/src/sensor_jepa): Core code for Sensor-JEPA, data loaders, temporal tokenization, world models, linear/MLP probes, and evaluations.
* [src/visual_jepa/](file:///c:/Users/Álvaro%20Schwiedop/Desktop/KriptaStudios/industrial_jepa_mvp/src/visual_jepa): Core code for Visual-JEPA, image-patching, and anomaly scoring.
* [scripts/](file:///c:/Users/Álvaro%20Schwiedop/Desktop/KriptaStudios/industrial_jepa_mvp/scripts): Execution scripts for pretraining, benchmarking, hard splits, and baseline evaluations.
* [configs/](file:///c:/Users/Álvaro%20Schwiedop/Desktop/KriptaStudios/industrial_jepa_mvp/configs): YAML configurations defining parameters for sensor and visual models.
* [tests/](file:///c:/Users/Álvaro%20Schwiedop/Desktop/KriptaStudios/industrial_jepa_mvp/tests): Extensive unit tests covering data shapes, leakage checks, metrics, and mask generation.
* [outputs/](file:///c:/Users/Álvaro%20Schwiedop/Desktop/KriptaStudios/industrial_jepa_mvp/outputs): Generated benchmarks, reports, and visualization outputs.

## 4. Implemented Modules and What Works
* **Sensor-JEPA Global Representation**: Trains Conv1D encoder/predictor with variance regularization (VICReg-style) to prevent collapse.
* **Sensor Action-Conditioned World Model**: Predicts future latent states $z_{t+h}$ given context actions (holder length, feed rate, hard/soft material, depth of cut, speed).
* **Failure Probing Scaffold**: Attaches linear/MLP classifiers on top of current or predicted future embeddings to classify if tool wear is critical.
* **Adversarial & Incremental Benchmarks**: Tools that calculate incremental performance metrics over simple metadata baselines.
* **Visual-JEPA MVP**: A small convolutional global encoder with masked latent prediction and heatmaps generation.
* **DenseSensorJEPA (Experimental)**: Tokenizes signal windows into temporal patches for patch-level target predictions.

## 5. What Does Not Work or Underperforms
* **Reusable Frozen Probes**: Probes trained on frozen JEPA features perform poorly (F1 macro $\sim 0.33-0.34$), suggesting the representation is not directly reusable without fine-tuning.
* **Global Visual-JEPA**: The global visual encoder is weak. Its anomaly AUROC ($\sim 0.67$ on MVTec bottle) fails to beat a basic pixel-stat baseline ($\sim 0.80$). Heatmaps are coarse.
* **Official Baselines Missing**: Because `sktime` and `aeon` are not installed locally, MiniROCKET/MultiROCKET baselines are fallback "lite" versions, and the TS2Vec baseline is a simple proxy, preventing strict SOTA eligibility.
* **DenseSensorJEPA Surprise**: The local surprise metrics do not yet outperform simple metadata/engineered baselines.

## 6. Primary CLI Commands
* Manifest creation: `python scripts/00_create_dataset_manifest.py`
* Run quick sensor demo: `python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml`
* Run quick visual demo: `python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml`
* Run all demos: `python scripts/17_run_all_demos.py`
* Run tests: `python -m pytest -q`

## 7. Honest Conclusion
We reject any claim that the current models achieve SOTA or beat all standard baselines. Adversarial auditing shows that **metadata/cycle-position proxies explain the majority of classification performance** under operational protocol. When cycle proxies are strictly removed (in "no-cycle" and "hard-generalization" splits), physical sensor signals and latent features do show incremental value over metadata, but Sensor-JEPA does not consistently outperform classical engineered features. Visual-JEPA requires a dense, patch-token level redesign. The project is an informative MVP that highlights the danger of proxy leakage and establishes a rigorous framework for true signal validation.
