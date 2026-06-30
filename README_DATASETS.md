# Dataset Status

Local data root: `data/raw`.

## Sensor

| Dataset | Local path | Status | Notes |
|---|---|---|---|
| cnc_milling | `data/raw/sensor/cnc_milling` | implemented | Main Sensor-JEPA MVP. Uses `FeatureAndMetadata_Milling.csv` and metadata. Raw CSV reader is optional because files are large. |
| cwru_bearing | `data/raw/sensor/cwru_bearing` | partial | Manifest and basic MAT inspection support. Not default training target. |
| paderborn_bearing | `data/raw/sensor/paderborn_bearing` | partial | Manifest and basic MAT inspection support. Not default training target. |
| multi_sensor_cnc | `data/raw/sensor/multi_sensor_cnc` | missing/pending | No local files in Fase 1 inventory. |
| nasa_ims_bearing | `data/raw/sensor/nasa_ims_bearing` | missing/pending | No local files in Fase 1 inventory. |
| cmapss | `data/raw/sensor/cmapss` | missing/pending | No local files in Fase 1 inventory. |
| ai4i | `data/raw/sensor/ai4i` | missing/pending | No local files in Fase 1 inventory. |

## Visual

| Dataset | Local path | Status | Notes |
|---|---|---|---|
| mvtec_ad | `data/raw/visual/mvtec_ad` | implemented | Main Visual-JEPA MVP. Train normals, test normals plus defects, masks if present. |
| visa | `data/raw/visual/visa` | partial | Manifest support through split CSV and category annotations. |
| kolektor_sdd | `data/raw/visual/kolektor_sdd` | partial | Manifest support for JPG/BMP mask pairs. |
| mvtec_3d_ad | `data/raw/visual/mvtec_3d_ad` | missing/pending | No local files in Fase 1 inventory. |
| neu_surface | `data/raw/visual/neu_surface` | missing/pending | No local files in Fase 1 inventory. |
| dagm_2007 | `data/raw/visual/dagm_2007` | missing/pending | No local files in Fase 1 inventory. |
| severstal | `data/raw/visual/severstal` | missing/pending | No local files in Fase 1 inventory. |
| wood_surface_defects | `data/raw/visual/wood_surface_defects` | missing/pending | No local files in Fase 1 inventory. |

Regenerate the machine-readable inventory:

```powershell
python scripts/00_create_dataset_manifest.py
```

Outputs:

- `data/manifests/datasets.yaml`
- `data/manifests/datasets.csv`

