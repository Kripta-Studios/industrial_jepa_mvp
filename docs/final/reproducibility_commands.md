# Index of Reproducibility Commands

This document contains a comprehensive index of PowerShell commands required to run, evaluate, benchmark, and verify all components of the `industrial_jepa_mvp` project.

---

## 1. Initial Setup and Compilation Checks

### Check Syntax and Compilability
```powershell
python -m compileall -q src scripts
```

### Run Python Unit Tests
```powershell
python -m pytest -q
```

---

## 2. Sensor-JEPA and World Model Commands

### Create Dataset Manifests
Generates the datasets YAML and CSV lists under `data/manifests/`.
```powershell
python scripts/00_create_dataset_manifest.py
```

### Sensor-JEPA Pretraining and Wear Probes Demo
Runs the quick end-to-end Sensor-JEPA pipeline (representation learning, linear and MLP probes, and basic CNN/GRU classifiers).
```powershell
python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml
```

### Sensor World Model (LeWorldModel) Training
Trains the action-conditioned world model to predict future latent states.
```powershell
python scripts/20_train_sensor_world_model.py --config configs/sensor_jepa/demo_sensor_quick.yaml
```

### Baseline Sensor Model Benchmark
Compares standard classifiers (Logistic Regression, RF, GBT, MLP) against the trained representation.
```powershell
python scripts/18_benchmark_sensor_models.py --config configs/sensor_jepa/demo_sensor_quick.yaml --quick
```

### SOTA-Candidate Sensor World Benchmark
Runs the multi-seed world model forecasting benchmark.
```powershell
# Quick smoke check
python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --quick

# Focused 3-seed validation run for h=3, K=10
python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --seeds 42,123,999 --horizons 3 --targets 10 --out-root outputs/sensor_jepa/sota_benchmark_h3_k10_3seed

# Adversarial 3-seed grid run across seeds, horizons (1,3,5,10,20) and targets (5,10,20)
python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --seeds 42,123,999 --horizons 1,3,5,10,20 --targets 5,10,20 --out-root outputs/sensor_jepa/sota_benchmark_adversarial_3seed
```

### Incremental Value Evaluation
Runs the matched-pair evaluation of sensor/metadata configurations and computes deltas.
```powershell
python scripts/22_incremental_sensor_value_benchmark.py --config configs/sensor_jepa/incremental_value_cnc.yaml
```

### Sensor Hard Generalization
Evaluates model performance under splits where tool IDs or cutting conditions are completely held out from training.
```powershell
python scripts/23_sensor_hard_generalization.py --config configs/sensor_jepa/hard_generalization_cnc.yaml
```

### JEPA vs Engineered Sensor Features Value Analysis
Runs comparative seed/horizon/target grids comparing raw features, engineered indicators, current embeddings, and predicted future embeddings.
```powershell
python scripts/35_jepa_vs_engineered_value.py --config configs/sensor_jepa/incremental_value_cnc.yaml --seeds 42,123,999 --horizons 1,3,5 --targets 5,10,20 --out-root outputs/sensor_jepa/jepa_vs_engineered_value
```

### Official Baseline Availability Auditor
Checks if library baselines (aeon, sktime, ts2vec) are installed.
```powershell
python scripts/24_install_check_sensor_baselines.py
```

### DenseSensorJEPA Pretraining & Evaluation
```powershell
# Pretrain DenseSensorJEPA
python scripts/25_pretrain_dense_sensor_jepa.py --config configs/sensor_jepa/dense_sensor_cnc.yaml

# Evaluate local surprise curves
python scripts/26_eval_dense_sensor_surprise.py --config configs/sensor_jepa/dense_sensor_cnc.yaml

# Train token-level world model
python scripts/27_train_token_sensor_world_model.py --config configs/sensor_jepa/dense_sensor_cnc.yaml
```

---

## 3. Visual-JEPA Commands

### Visual-JEPA Global MVP Demo
Trains a global convolutional Visual-JEPA on normal MVTec images and evaluates reconstruction/anomaly detection on the test set.
```powershell
python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml
```

### Run All Demos (Sensor + Visual Global MVP)
```powershell
python scripts/17_run_all_demos.py
```

### Visual Anomaly Model Benchmarks
```powershell
python scripts/19_benchmark_visual_models.py --config configs/visual_jepa/demo_visual_quick.yaml --quick
```

---

## 4. Academic Paper and Presentation Checks

### Compile Academic LaTeX Paper
```powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```
*(Note: If LaTeX compilation tools are not installed in the target workspace shell, review files directly.)*

### View Commercial HTML Presentation
Locate the HTML presentation in your file explorer:
`presentation/mvp_galicia.html`
Right-click the file and select "Open with" and choose a modern browser (Chrome, Edge, Firefox, Safari) to view the interactive slides.
