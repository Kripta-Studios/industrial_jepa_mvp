# Reproducibility Commands

## Manifest

```powershell
python scripts/40_build_iwm_manifest.py
```

## Visual Foundation

```powershell
python scripts/41_visual_foundation_benchmark.py --config configs/industrial_world_model/visual_foundation_mvtec.yaml
python scripts/41_visual_foundation_benchmark.py --config configs/industrial_world_model/visual_foundation_visa.yaml
python scripts/41_visual_foundation_benchmark.py --config configs/industrial_world_model/visual_foundation_kolektor.yaml
```

## Feature Extraction

```powershell
python scripts/42_extract_visual_features.py --config configs/industrial_world_model/visual_foundation_mvtec.yaml
```

## Heatmaps

```powershell
python scripts/43_visual_anomaly_heatmaps.py --config configs/industrial_world_model/visual_foundation_mvtec.yaml
```

## LeJEPA Visual

```powershell
python scripts/44_pretrain_lejepa_visual.py --config configs/industrial_world_model/lejepa_visual_mvtec.yaml
python scripts/45_eval_lejepa_visual.py --config configs/industrial_world_model/lejepa_visual_mvtec.yaml
python scripts/46_lejepa_probe_benchmark.py --config configs/industrial_world_model/lejepa_visual_mvtec.yaml
```

## Sensor/Process

```powershell
python scripts/47_pretrain_lejepa_sensor.py --config configs/industrial_world_model/lejepa_sensor_cnc.yaml
python scripts/48_eval_lejepa_sensor.py --config configs/industrial_world_model/lejepa_sensor_cnc.yaml
```

## LeWorldModel

```powershell
python scripts/49_pretrain_leworldmodel_sensor.py --config configs/industrial_world_model/leworldmodel_cnc.yaml
python scripts/50_eval_leworldmodel_sensor.py --config configs/industrial_world_model/leworldmodel_cnc.yaml
python scripts/53_leworldmodel_surprise_benchmark.py --config configs/industrial_world_model/leworldmodel_cnc.yaml
```

## Hierarchy

```powershell
python scripts/54_hierarchical_iwm_benchmark.py
python scripts/55_hierarchical_iwm_report.py
```

## Product Demo

```powershell
python scripts/56_build_product_demo.py
```

## Validation

```powershell
python -m compileall -q src scripts
python -m pytest -q
```
