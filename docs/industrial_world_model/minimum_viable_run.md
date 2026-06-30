# Minimum Viable Run

This run is designed to finish quickly on CPU.

```powershell
python scripts/40_build_iwm_manifest.py
python scripts/41_visual_foundation_benchmark.py --config configs/industrial_world_model/visual_foundation_mvtec.yaml --quick --categories bottle --max-samples 80
python scripts/44_pretrain_lejepa_visual.py --config configs/industrial_world_model/lejepa_visual_mvtec.yaml --quick --epochs 2 --categories bottle
python scripts/47_pretrain_lejepa_sensor.py --config configs/industrial_world_model/lejepa_sensor_cnc.yaml --quick --epochs 2
python scripts/48_eval_lejepa_sensor.py --config configs/industrial_world_model/lejepa_sensor_cnc.yaml --quick
python scripts/49_pretrain_leworldmodel_sensor.py --config configs/industrial_world_model/leworldmodel_cnc.yaml --quick --epochs 2
python scripts/53_leworldmodel_surprise_benchmark.py --config configs/industrial_world_model/leworldmodel_cnc.yaml
python scripts/54_hierarchical_iwm_benchmark.py
python scripts/56_build_product_demo.py
python -m pytest -q
```

Expected outputs:

- `outputs/industrial_world_model/dataset_manifest.json`
- `outputs/industrial_world_model/visual_foundation/results.csv`
- `outputs/industrial_world_model/lejepa_visual/pretrain_log.csv`
- `outputs/industrial_world_model/sensor_lejepa/results.csv`
- `outputs/industrial_world_model/leworldmodel/surprise_results.csv`
- `outputs/industrial_world_model/hierarchy/top_alerts.csv`
- `product_demo/industrial_world_model/index.html`
