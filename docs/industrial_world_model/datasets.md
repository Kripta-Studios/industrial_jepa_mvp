# Datasets

The manifest script checks local availability and does not download data automatically.

Supported local paths:

- `data/raw/visual/mvtec_ad/`
- `data/raw/visual/visa/`
- `data/raw/visual/kolektor_sdd/`
- `data/raw/sensor/cnc_milling/`
- `data/raw/sensor/cwru_bearing/`
- `data/raw/sensor/paderborn_bearing/`

Optional datasets marked as missing when absent:

- MVTec AD 2;
- MVTec LOCO AD;
- Real-IAD;
- MVTec 3D-AD;
- NEU Surface Defect;
- DAGM 2007;
- Severstal;
- Wood Surface Defects;
- Tennessee Eastman Process;
- NASA C-MAPSS;
- DROID;
- BridgeData V2.

Run:

```powershell
python scripts/40_build_iwm_manifest.py
```
