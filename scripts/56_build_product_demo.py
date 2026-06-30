from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return [{"status": "missing", "path": str(path)}]
    df = pd.read_csv(path)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Industrial World Model product demo.")
    parser.add_argument("--out-root", default="product_demo/industrial_world_model")
    parser.add_argument("--results-root", default="outputs/industrial_world_model")
    args = parser.parse_args()
    out = Path(args.out_root)
    res = Path(args.results_root)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "title": "Industrial Predictive Quality World Model",
        "positioning": "Pilot-ready research MVP for predictive quality and anomaly detection.",
        "dataset_manifest": _read_csv(res / "dataset_manifest.csv") if (res / "dataset_manifest.csv").exists() else json.loads((res / "dataset_manifest.json").read_text(encoding="utf-8")) if (res / "dataset_manifest.json").exists() else [{"status": "missing"}],
        "visual_foundation": _read_csv(res / "visual_foundation" / "results.csv"),
        "lejepa_visual": _read_csv(res / "lejepa_visual" / "anomaly_results.csv"),
        "sensor_lejepa": _read_csv(res / "sensor_lejepa" / "results.csv"),
        "leworldmodel": _read_csv(res / "leworldmodel" / "surprise_results.csv"),
        "hierarchy": _read_csv(res / "hierarchy" / "top_alerts.csv"),
    }
    (out / "demo_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Industrial Predictive Quality World Model</title>
  <style>
    :root { --bg:#0b1220; --panel:#121b2e; --text:#f7fafc; --muted:#a7b0c0; --accent:#38bdf8; --ok:#22c55e; }
    body { margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--text); }
    header, section { max-width:1120px; margin:auto; padding:56px 24px; }
    .hero { min-height:55vh; display:grid; align-items:center; }
    h1 { font-size: clamp(2.2rem, 5vw, 4.8rem); line-height:1; margin:0 0 20px; }
    h2 { font-size:2rem; margin:0 0 18px; }
    p { color:var(--muted); font-size:1.05rem; line-height:1.6; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:16px; }
    .card { background:var(--panel); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:20px; }
    .metric { font-size:2rem; color:var(--accent); font-weight:800; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border-radius:12px; overflow:hidden; }
    th,td { padding:10px 12px; border-bottom:1px solid rgba(255,255,255,.08); text-align:left; font-size:.92rem; }
    th { color:#dbeafe; }
    .badge { display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(56,189,248,.14); color:#bae6fd; margin-right:8px; }
  </style>
</head>
<body>
  <header class="hero">
    <div>
      <span class="badge">Vision + Sensors + Process</span><span class="badge">Pilot-ready MVP</span>
      <h1>Industrial Predictive Quality World Model</h1>
      <p>Un sistema para aprender el comportamiento esperado de procesos industriales normales y priorizar desviaciones: defectos visuales, señales sensoriales anómalas y riesgo por ciclo/lote/línea.</p>
    </div>
  </header>
  <section><h2>Qué entrega el MVP</h2><div class="grid">
    <div class="card"><div class="metric">Visual</div><p>DINO/ResNet-style dense features, PatchCore-lite, PaDiM-lite y heatmaps de defecto.</p></div>
    <div class="card"><div class="metric">LeJEPA</div><p>SIGReg e in-domain pretraining como módulo experimental medido contra baselines fuertes.</p></div>
    <div class="card"><div class="metric">World Model</div><p>Predicción latente z_t + acciones/setpoints -> z_t+h y surprise para predictive quality.</p></div>
    <div class="card"><div class="metric">Jerarquía</div><p>Agregación patch -> imagen/pieza -> ciclo -> lote -> línea con ranking de inspección.</p></div>
  </div></section>
  <section><h2>Resultados cargados</h2><div id="tables"></div></section>
  <section><h2>Piloto comercial</h2><p>El piloto dura 4-6 semanas. El cliente aporta imágenes, sensores, metadatos, setpoints y registros de calidad/fallo. Se entrega benchmark, demo local, heatmaps, ranking top-10 de alertas y recomendación go/no-go.</p></section>
  <section><h2>Limitaciones honestas</h2><p>No se presenta como plataforma de producción. No afirma causalidad ni control autónomo. Las acciones/setpoints reales son necesarias para validar predictive quality fuerte.</p></section>
  <script type="application/json" id="demo-data">__DEMO_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('demo-data').textContent);
    const root=document.getElementById('tables');
    for (const [name, rows] of Object.entries(data)) {
      if (!Array.isArray(rows)) continue;
      const card=document.createElement('div'); card.className='card'; card.style.marginBottom='16px';
      card.innerHTML='<h3>'+name+'</h3>';
      const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))].slice(0,8);
      let html='<table><thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>';
      html+=rows.slice(0,8).map(r=>'<tr>'+keys.map(k=>'<td>'+(r[k]??'')+'</td>').join('')+'</tr>').join('');
      html+='</tbody></table>'; card.innerHTML+=html; root.appendChild(card);
    }
  </script>
</body>
</html>
"""
    html = html.replace("__DEMO_DATA__", json.dumps(data).replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "README.md").write_text("# Industrial World Model Demo\n\nOpen `index.html` locally. The page embeds the generated data so it works by double click, and `demo_data.json` is kept next to it for reproducibility/audit.\n", encoding="utf-8")
    print(f"Demo written to {out / 'index.html'}")


if __name__ == "__main__":
    main()
