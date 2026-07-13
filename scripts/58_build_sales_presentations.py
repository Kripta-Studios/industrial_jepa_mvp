from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
]


CONTACTS = [
    ("FINSA", "finsa@finsa.es", "+34 981 050 000", "Madera / tableros", "Surface quality, process sensors and predictive maintenance in production lines."),
    ("Televés", "televes@televes.com", "981 52 22 00", "Electrónica", "Test logs, PCB inspection, production anomaly detection and quality control."),
    ("Norvento", "norvento@norvento.com", "+34 982 227889", "Energía", "SCADA, vibration, early alerts and predictive maintenance for energy assets."),
    ("Marine Instruments", "marineinstruments@marineinstruments.es", "986366360", "Telemetría marina", "Marine telemetry, sensors and early anomaly detection in equipment fleets."),
    ("Alén Space", "info@alen.space", "", "Aeroespacial", "Satellite telemetry, sensor alerts and operational anomaly ranking."),
    ("Cinfo", "Formulario web", "+34 881 896 975", "Vídeo / visión", "Video analytics, visual anomaly detection, events and industrial vision pilots."),
]


T = {
    "es": {
        "lang": "es",
        "title": "MVP de IA Industrial para Calidad Predictiva",
        "nav": ["Producto", "Resultados", "Piloto", "Sectores", "Contacto"],
        "hero_badges": ["Visión + sensores", "Resultados post-audit pendientes", "Risk scoring", "Piloto 4-6 semanas"],
        "hero_title": "Industrial Predictive Quality MVP",
        "hero_subtitle": "Anticipar defectos, fallos y desviaciones combinando visión industrial, sensores y datos de proceso.",
        "hero_body": "Un piloto rápido para transformar datos históricos de planta en rankings de riesgo, heatmaps de defecto y decisiones de inspección accionables.",
        "cta_primary": "Ver producto",
        "cta_secondary": "Plan de piloto",
        "problem_title": "El problema de negocio",
        "problem_subtitle": "Las plantas ya generan datos; el reto es convertirlos en decisiones antes de que aparezca el coste.",
        "problem_cards": [
            ("Paradas no planificadas", "Riesgo operativo, retrasos, scrap y mantenimiento reactivo."),
            ("Defectos visuales tardíos", "Inspección manual costosa y defectos detectados demasiado tarde."),
            ("Señales infrautilizadas", "SCADA, PLC, test logs, cámaras y GMAO suelen analizarse por separado."),
            ("Demasiadas alarmas", "Los equipos necesitan priorización, no más ruido."),
        ],
        "product_title": "Qué vendemos como MVP",
        "product_subtitle": "Un sistema modular para priorizar inspecciones, anticipar riesgo de fallo y convertir datos industriales en decisiones accionables.",
        "modules": [
            ("Visual Quality Foundation", "DINOv2 preentrenado, PatchCore-lite, PaDiM-lite, heatmaps y ranking de imágenes/piezas."),
            ("Machine Failure Risk Scoring", "Sensores físicos, variables de proceso, features engineered, modelos calibrados y alertas top 5-10%."),
            ("Process Intelligence", "Metadatos, setpoints, recetas, test logs y contexto operativo integrados en un único score."),
            ("Predictive Quality World Model", "Modelo de evolución esperada del proceso para detectar desviaciones tempranas con secuencias y setpoints."),
        ],
        "capabilities_title": "Tres productos en un mismo piloto",
        "capabilities_subtitle": "El cliente recibe valor aunque cada línea tenga datos distintos: cámaras, sensores, SCADA, test logs o mantenimiento.",
        "capabilities": [
            ("Visual defect detection", "Heatmaps sobre pieza/imagen y ranking de inspección para superficie, PCB, envases, producto o vídeo."),
            ("Machine failure risk", "Predicción de fallo próximo con sensores, ciclo, proceso y lead time operativo para mantenimiento."),
            ("Process deviation alerts", "Comparación entre evolución esperada y observada para avisar antes de scrap, parada o pérdida de calidad."),
        ],
        "model_stack_title": "Modelos integrados en el MVP",
        "model_stack": [
            ("DINOv2 / ResNet", "features densas visuales"),
            ("PatchCore / PaDiM", "memoria normal + heatmaps"),
            ("Engineered sensor features", "RMS, energía, frecuencia, kurtosis"),
            ("Latent future features", "señal aprendida opcional"),
            ("World-model surprise", "desviación temporal esperada"),
            ("Hierarchical ranking", "patch → pieza → ciclo → lote → línea"),
        ],
        "diagram_title": "Arquitectura del piloto",
        "diagram_nodes": ["Datos planta", "Señal útil", "Modelos", "Alertas", "Decisión"],
        "diagram_details": [
            "Cámaras, sensores, SCADA, test logs, GMAO",
            "Visión densa, memoria visual, sensores físicos",
            "Risk scoring, anomalías y desviación esperada",
            "Top 10%, heatmaps, pieza/ciclo/lote/línea",
            "Inspeccionar, ajustar e integrar",
        ],
        "evidence_title": "Estado post-auditoría de la evidencia",
        "evidence_subtitle": "Las cifras visuales con umbral elegido en test y las de dureza con valores físicos solapados quedan retiradas hasta rerun.",
        "visual_evidence": "Visual MVTec bottle quick",
        "sensor_evidence": "Sensor CNC no-cycle",
        "hardness_evidence": "Hardness/material split",
        "business_reading": "Lectura comercial",
        "reading_text": "Cuando no basta con contar ciclos o revisar umbrales, las señales físicas y visuales aportan información accionable. El piloto valida si eso ocurre en los datos del cliente.",
        "results": [
            ("Visual thresholded", "RETIRADO", "pre-fix", "Umbral ajustado en test; rerun validation-only pendiente"),
            ("DINO / fallback", "PENDIENTE", "trazabilidad", "Solo actual_backbone verificado puede nombrarse"),
            ("Held-out hardness", "RETIRADO", "pre-fix", "Valores físicos solapados; rerun disjunto pendiente"),
            ("Resultados post-audit", "0", "runs", "results/index.json no registra aún un benchmark corregido"),
        ],
        "machine_title": "Módulo específico: fallo próximo en máquina",
        "machine_subtitle": "La parte sensor no se vende como magia: se vende como scoring de riesgo con señales físicas, proceso y alertas priorizadas.",
        "machine_kpis": [
            ("Estado", "SMOKE", "Capacidad de pipeline; no benchmark post-audit"),
            ("Threshold", "VAL", "Se ajusta solo en validación"),
            ("Hardness", "RAW", "Valores físicos disjuntos y registrados"),
            ("Backbone", "ACTUAL", "Fallback explícito o fallo cerrado"),
        ],
        "services_title": "Producto y servicios ofrecidos",
        "services": [
            ("Piloto visual", "Calidad de superficie, PCB, envases, producto y vídeo industrial con heatmaps."),
            ("Piloto fallo máquina", "Risk scoring con vibración, corrientes, telemetría, SCADA, ciclo, proceso y lead time."),
            ("Auditoría de datos", "Qué señal sirve, qué variables son redundantes y qué se puede automatizar."),
            ("Demo ejecutiva", "HTML/PDF, tablas, rankings de inspección y roadmap de integración."),
        ],
        "contacts_title": "Cuentas prioritarias en Galicia",
        "contacts_subtitle": "Contactos públicos para outreach. No implica relación, interés ni partnership.",
        "pilot_title": "Piloto de 4-6 semanas",
        "pilot_steps": [
            ("1. Ingesta", "Datos históricos, imágenes, sensores, logs y eventos de calidad/fallo."),
            ("2. Baselines", "Reglas simples, modelos clásicos y baseline visual fuerte."),
            ("3. Modelado", "Risk scoring, anomaly detection, heatmaps y ranking top 10%."),
            ("4. Decisión", "Informe de inversión, demo local y plan de integración."),
        ],
        "deliverables_title": "Entregables del piloto",
        "deliverables": [
            "Benchmark con métricas reales por categoría o activo.",
            "Heatmaps visuales y ranking top de inspecciones.",
            "Score de riesgo por pieza, ciclo, lote o línea cuando existan datos.",
            "Informe ejecutivo y técnico con recomendaciones.",
            "Demo HTML local y PDF para dirección.",
        ],
        "scope_title": "Alcance claro para vender con confianza",
        "scope_left": "Lo que sí entregamos",
        "scope_right": "Condiciones para desplegar",
        "scope_yes": [
            "Validación con datos reales del cliente.",
            "Comparación contra baselines simples y fuertes.",
            "Métricas útiles: AUPRC, Precision@10, falsas alarmas y lead time.",
            "Producto modular: visión, sensores, proceso y alertas.",
        ],
        "scope_conditions": [
            "Datos etiquetados o eventos históricos suficientes.",
            "Revisión con expertos de planta.",
            "Piloto antes de integración operativa.",
            "Monitorización y mantenimiento si pasa a producción.",
        ],
        "footer": "Proyecto industrial_jepa_mvp · MVP comercial de IA industrial",
        "meeting_title": "MVP de IA Industrial",
        "meeting_subtitle": "Calidad predictiva, anomalías visuales y risk scoring para empresas gallegas.",
        "next_step": "Próximo paso: llamada técnica de 15 minutos y selección de una línea/caso piloto.",
    },
    "en": {
        "lang": "en",
        "title": "Industrial AI MVP for Predictive Quality",
        "nav": ["Product", "Results", "Pilot", "Sectors", "Contact"],
        "hero_badges": ["Vision + sensors", "Post-audit results pending", "Risk scoring", "4-6 week pilot"],
        "hero_title": "Industrial Predictive Quality MVP",
        "hero_subtitle": "Anticipate defects, failures and process deviations by combining industrial vision, sensors and process data.",
        "hero_body": "A fast pilot to turn plant history into risk rankings, defect heatmaps and actionable inspection decisions.",
        "cta_primary": "View product",
        "cta_secondary": "Pilot plan",
        "problem_title": "The business problem",
        "problem_subtitle": "Plants already generate data; the challenge is converting it into decisions before the cost appears.",
        "problem_cards": [
            ("Unplanned downtime", "Operational risk, delays, scrap and reactive maintenance."),
            ("Late visual defects", "Manual inspection is expensive and defects are detected too late."),
            ("Underused signals", "SCADA, PLC, test logs, cameras and maintenance systems are usually analysed separately."),
            ("Too many alarms", "Engineering teams need prioritisation, not more noise."),
        ],
        "product_title": "What we sell as an MVP",
        "product_subtitle": "A modular system to prioritise inspections, anticipate machine failure risk and turn industrial data into actionable decisions.",
        "modules": [
            ("Visual Quality Foundation", "Pretrained DINOv2, PatchCore-lite, PaDiM-lite, heatmaps and image/part ranking."),
            ("Machine Failure Risk Scoring", "Physical sensors, process variables, engineered features, calibrated models and top 5-10% alerts."),
            ("Process Intelligence", "Metadata, setpoints, recipes, test logs and operational context integrated into one score."),
            ("Predictive Quality World Model", "Expected-process-evolution model to detect early deviations from sequences and setpoints."),
        ],
        "capabilities_title": "Three products in one pilot",
        "capabilities_subtitle": "The customer receives value across different data realities: cameras, sensors, SCADA, test logs or maintenance history.",
        "capabilities": [
            ("Visual defect detection", "Part/image heatmaps and inspection ranking for surfaces, PCB, packaging, products or video."),
            ("Machine failure risk", "Failure-soon prediction using sensors, cycle, process data and operational lead time for maintenance."),
            ("Process deviation alerts", "Expected vs observed process evolution to warn before scrap, downtime or quality loss."),
        ],
        "model_stack_title": "Models integrated in the MVP",
        "model_stack": [
            ("DINOv2 / ResNet", "dense visual features"),
            ("PatchCore / PaDiM", "normal memory + heatmaps"),
            ("Engineered sensor features", "RMS, energy, frequency, kurtosis"),
            ("Latent future features", "optional learned signal"),
            ("World-model surprise", "expected temporal deviation"),
            ("Hierarchical ranking", "patch → part → cycle → batch → line"),
        ],
        "diagram_title": "Pilot architecture",
        "diagram_nodes": ["Plant data", "Useful signal", "Models", "Alerts", "Decision"],
        "diagram_details": [
            "Cameras, sensors, SCADA, test logs, maintenance records",
            "Dense vision, visual memory, physical sensors",
            "Risk scoring, anomalies and expected deviation",
            "Top 10%, heatmaps, part/cycle/batch/line",
            "Inspect, adjust and integrate",
        ],
        "evidence_title": "Post-audit evidence status",
        "evidence_subtitle": "Visual test-fitted thresholds and hardness splits with overlapping physical values are withdrawn pending rerun.",
        "visual_evidence": "Visual MVTec bottle quick",
        "sensor_evidence": "Sensor CNC no-cycle",
        "hardness_evidence": "Hardness/material split",
        "business_reading": "Business reading",
        "reading_text": "When cycle counts or threshold checks are not enough, physical and visual signals can add actionable information. The pilot validates whether this is true in the client's own data.",
        "results": [
            ("Visual thresholded", "WITHDRAWN", "pre-fix", "Test-fitted threshold; validation-only rerun pending"),
            ("DINO / fallback", "PENDING", "provenance", "Only verified actual_backbone may be named"),
            ("Held-out hardness", "WITHDRAWN", "pre-fix", "Physical-value overlap; disjoint rerun pending"),
            ("Post-audit results", "0", "runs", "results/index.json contains no corrected benchmark yet"),
        ],
        "machine_title": "Specific module: machine failure-soon risk",
        "machine_subtitle": "The sensor module is sold as risk scoring: physical signals, process context and prioritised alerts.",
        "machine_kpis": [
            ("Status", "SMOKE", "Pipeline capability; no post-audit benchmark"),
            ("Threshold", "VAL", "Fit only on validation"),
            ("Hardness", "RAW", "Physical values disjoint and recorded"),
            ("Backbone", "ACTUAL", "Explicit fallback or fail closed"),
        ],
        "services_title": "Product and services",
        "services": [
            ("Visual pilot", "Surface quality, PCB, packaging, product and industrial video with heatmaps."),
            ("Machine failure pilot", "Risk scoring with vibration, currents, telemetry, SCADA, cycle, process data and lead time."),
            ("Data audit", "Which signals work, which variables are redundant and what can be automated."),
            ("Executive demo", "HTML/PDF, tables, inspection rankings and integration roadmap."),
        ],
        "contacts_title": "Priority accounts in Galicia",
        "contacts_subtitle": "Public outreach contacts. This does not imply a relationship, interest or partnership.",
        "pilot_title": "4-6 week pilot",
        "pilot_steps": [
            ("1. Ingestion", "Historical data, images, sensors, logs and quality/failure events."),
            ("2. Baselines", "Simple rules, classical models and strong visual baselines."),
            ("3. Modelling", "Risk scoring, anomaly detection, heatmaps and top 10% ranking."),
            ("4. Decision", "Investment report, local demo and integration plan."),
        ],
        "deliverables_title": "Pilot deliverables",
        "deliverables": [
            "Benchmark with real metrics by category or asset.",
            "Visual heatmaps and top inspection ranking.",
            "Risk score by part, cycle, batch or line when data exists.",
            "Executive and technical report with recommendations.",
            "Local HTML demo and PDF for management.",
        ],
        "scope_title": "Clear scope for confident selling",
        "scope_left": "What we deliver",
        "scope_right": "Conditions for deployment",
        "scope_yes": [
            "Validation on the client's real data.",
            "Comparison against simple and strong baselines.",
            "Useful metrics: AUPRC, Precision@10, false alarms and lead time.",
            "Modular product: vision, sensors, process and alerts.",
        ],
        "scope_conditions": [
            "Enough labelled data or historical events.",
            "Review with plant experts.",
            "Pilot before operational integration.",
            "Monitoring and maintenance if it moves to production.",
        ],
        "footer": "industrial_jepa_mvp project · Commercial industrial AI MVP",
        "meeting_title": "Industrial AI MVP",
        "meeting_subtitle": "Predictive quality, visual anomalies and risk scoring for Galician companies.",
        "next_step": "Next step: 15-minute technical call and selection of one pilot line/use case.",
    },
}


def pct_width(value: str) -> str:
    try:
        return f"{min(float(value) * 100.0, 100.0):.0f}%"
    except ValueError:
        return "0%"


def base_css(deck: bool = False) -> str:
    return f"""
    :root {{
      --bg:#f5f7fb; --panel:#ffffff; --ink:#172033; --muted:#64748b; --line:#dbe3ef;
      --blue:#0b6bcb; --blue2:#e8f2ff; --green:#16804a; --amber:#b76b00;
      --shadow:0 12px 34px rgba(15,23,42,.08);
    }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg); color:var(--ink); line-height:1.5; }}
    @page {{ size: letter; margin: 0; }}
    a {{ color:inherit; }}
    header.nav {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.88); backdrop-filter:blur(18px); border-bottom:1px solid var(--line); padding:14px 28px; display:flex; justify-content:space-between; align-items:center; }}
    .brand {{ font-weight:750; color:var(--ink); }}
    nav {{ display:flex; gap:18px; font-size:.92rem; color:var(--muted); }}
    .section {{ max-width:1160px; margin:0 auto; padding:72px 28px; }}
    .hero {{ min-height:72vh; display:grid; place-items:center; text-align:center; }}
    h1 {{ font-size:clamp(2.6rem,6vw,5.5rem); line-height:.98; letter-spacing:-.055em; margin:18px 0; }}
    h2 {{ font-size:clamp(2rem,4vw,3.35rem); line-height:1.05; letter-spacing:-.04em; margin-bottom:14px; }}
    h3 {{ font-size:1.25rem; margin-bottom:8px; }}
    .lead {{ color:var(--muted); font-size:1.28rem; max-width:880px; margin:0 auto 22px; }}
    .bodycopy {{ color:var(--muted); font-size:1.06rem; max-width:820px; margin:0 auto; }}
    .badges {{ display:flex; gap:9px; flex-wrap:wrap; justify-content:center; }}
    .badge {{ border:1px solid rgba(11,107,203,.18); background:var(--blue2); color:var(--blue); border-radius:999px; padding:7px 11px; font-size:.78rem; font-weight:750; text-transform:uppercase; letter-spacing:.04em; }}
    .actions {{ display:flex; gap:12px; justify-content:center; margin-top:28px; flex-wrap:wrap; }}
    .btn {{ border-radius:999px; padding:12px 20px; text-decoration:none; font-weight:700; border:1px solid var(--line); background:var(--panel); }}
    .btn.primary {{ background:var(--blue); color:white; border-color:var(--blue); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; width:100%; }}
    .grid.two {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .grid.three {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:22px; box-shadow:var(--shadow); }}
    .card p, .card li {{ color:var(--muted); }}
    .metric {{ font-size:2.25rem; font-weight:850; color:var(--blue); letter-spacing:-.04em; }}
    .metric.small {{ font-size:1.85rem; }}
    .caption {{ color:var(--muted); font-size:.92rem; }}
    .model-map {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-top:20px; }}
    .model-node {{ background:white; border:1px solid var(--line); border-radius:16px; padding:14px; min-height:118px; box-shadow:var(--shadow); }}
    .model-node strong {{ color:var(--blue); display:block; margin-bottom:6px; }}
    .value-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }}
    .kpi {{ background:linear-gradient(180deg,#fff,#f8fbff); border:1px solid var(--line); border-radius:16px; padding:18px; }}
    .kpi .metric {{ font-size:2rem; }}
    .pipeline {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:24px; }}
    .pipe {{ background:linear-gradient(180deg,#fff,#f7fbff); border:1px solid rgba(11,107,203,.15); border-radius:17px; padding:18px; min-height:150px; position:relative; }}
    .pipe:not(:last-child)::after {{ content:"→"; position:absolute; right:-16px; top:48%; color:var(--blue); font-weight:900; font-size:1.5rem; }}
    .pipe .num {{ width:28px; height:28px; display:grid; place-items:center; border-radius:999px; background:var(--blue); color:white; font-weight:800; margin-bottom:10px; }}
    .bars {{ display:grid; gap:12px; }}
    .bar-label {{ display:flex; justify-content:space-between; gap:10px; font-size:.92rem; margin-bottom:5px; }}
    .bar-bg {{ height:10px; background:#e5edf7; border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; background:linear-gradient(90deg,var(--blue),#41a5ff); border-radius:999px; }}
    .table {{ width:100%; border-collapse:collapse; background:var(--panel); border-radius:16px; overflow:hidden; box-shadow:var(--shadow); }}
    .table th,.table td {{ padding:12px 14px; border-bottom:1px solid var(--line); text-align:left; font-size:.95rem; }}
    .table th {{ background:#eef4fb; }}
    .contacts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}
    .contact {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; }}
    .contact .tag {{ color:var(--blue); font-size:.78rem; font-weight:800; text-transform:uppercase; }}
    .timeline {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
    .step {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:var(--shadow); }}
    .step strong {{ color:var(--blue); }}
    .split {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    ul.clean {{ list-style:none; display:grid; gap:10px; }}
    ul.clean li::before {{ content:"✓"; color:var(--green); font-weight:900; margin-right:8px; }}
    footer {{ padding:32px; text-align:center; color:var(--muted); border-top:1px solid var(--line); background:white; }}
    @media(max-width:950px) {{ .model-map{{grid-template-columns:repeat(2,1fr)}} }}
    @media(max-width:850px) {{ nav{{display:none}} .grid.two,.grid.three,.split,.pipeline,.timeline,.model-map{{grid-template-columns:1fr}} .pipe:not(:last-child)::after{{content:"↓"; right:50%; top:auto; bottom:-20px}} }}
    {deck_css() if deck else ""}
    """


def deck_css() -> str:
    return """
    body { overflow:hidden; height:100vh; }
    header.nav { position:fixed; width:100%; height:58px; }
    .deck { height:100vh; position:relative; overflow:hidden; padding-top:58px; }
    .slide { position:absolute; inset:58px 0 56px; padding:48px 64px; display:flex; flex-direction:column; justify-content:center; opacity:0; pointer-events:none; transform:translateX(40px); transition:.35s ease; background:var(--bg); }
    .slide.active { opacity:1; pointer-events:auto; transform:none; }
    .slide-inner { max-width:1120px; margin:0 auto; width:100%; }
    .controls { position:fixed; bottom:0; left:0; right:0; height:56px; display:flex; justify-content:space-between; align-items:center; padding:0 28px; background:white; border-top:1px solid var(--line); color:var(--muted); }
    .controls button { width:38px; height:38px; border-radius:999px; border:1px solid var(--line); background:#fff; cursor:pointer; }
    .dots { display:flex; gap:7px; }
    .dot { width:8px; height:8px; border-radius:99px; background:#cbd5e1; }
    .dot.active { background:var(--blue); transform:scale(1.25); }
    @media print {
      body { overflow:visible; height:auto; background:white; }
      header.nav,.controls { display:none; }
      .deck { height:auto; overflow:visible; padding-top:0; }
      .slide { position:relative; inset:auto; opacity:1; pointer-events:auto; transform:none; min-height:100vh; page-break-after:always; padding:42px 48px; background:white; }
      .slide:last-child { page-break-after:auto; }
      .card,.step,.contact,.table { box-shadow:none; }
    }
    """


def render_contacts(lang: str) -> str:
    rows = []
    for i, (name, email, phone, sector, value) in enumerate(CONTACTS, 1):
        channel = "Formulario / llamada" if name == "Cinfo" and lang == "es" else "Web form / call" if name == "Cinfo" else "Email"
        value_text = value
        if lang == "es":
            translations = {
                "Surface quality, process sensors and predictive maintenance in production lines.": "Calidad de superficie, sensores de proceso y mantenimiento predictivo en líneas industriales.",
                "Test logs, PCB inspection, production anomaly detection and quality control.": "Test logs, inspección visual de PCB, anomalía en producción y control de calidad.",
                "SCADA, vibration, early alerts and predictive maintenance for energy assets.": "SCADA, vibración, alertas tempranas y mantenimiento predictivo en activos energéticos.",
                "Marine telemetry, sensors and early anomaly detection in equipment fleets.": "Telemetría marina, sensores y anomalías tempranas en flotas de equipos.",
                "Satellite telemetry, sensor alerts and operational anomaly ranking.": "Telemetría satelital, alertas por sensores y ranking de riesgo operacional.",
                "Video analytics, visual anomaly detection, events and industrial vision pilots.": "Vídeo, anomalía visual, eventos y pilotos de visión industrial.",
            }
            value_text = translations[value]
        contact = email if not phone else f"{email} · {phone}"
        rows.append(f"""
        <div class="contact">
          <div class="tag">{i} · {channel}</div>
          <h3>{name}</h3>
          <p><strong>{sector}</strong></p>
          <p>{value_text}</p>
          <p class="caption">{contact}</p>
        </div>""")
    return "\n".join(rows)


def render_result_cards(tr: dict) -> str:
    return "\n".join(
        f"""<div class="card"><div class="metric">{value}</div><h3>{name}</h3><p>{metric} · {note}</p></div>"""
        for name, value, metric, note in tr["results"]
    )


def render_capabilities(tr: dict) -> str:
    return "\n".join(
        f"""<div class="card"><h3>{name}</h3><p>{text}</p></div>"""
        for name, text in tr["capabilities"]
    )


def render_model_stack(tr: dict) -> str:
    return "\n".join(
        f"""<div class="model-node"><strong>{name}</strong><p>{text}</p></div>"""
        for name, text in tr["model_stack"]
    )


def render_machine_kpis(tr: dict) -> str:
    return "\n".join(
        f"""<div class="kpi"><div class="metric small">{value}</div><h3>{name}</h3><p>{note}</p></div>"""
        for name, value, note in tr["machine_kpis"]
    )


def render_bars(tr: dict) -> str:
    return "\n".join(
        f"""<div>
          <div class="bar-label"><span>{name}</span><strong>{value} {metric}</strong></div>
          <div class="bar-bg"><div class="bar-fill" style="width:{pct_width(value)}"></div></div>
          <p class="caption">{note}</p>
        </div>"""
        for name, value, metric, note in tr["results"]
    )


def render_web(lang: str) -> str:
    tr = T[lang]
    nav = "".join(f'<a href="#s{i}">{label}</a>' for i, label in enumerate(tr["nav"], 1))
    badges = "".join(f'<span class="badge">{b}</span>' for b in tr["hero_badges"])
    problem = "".join(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>' for h, p in tr["problem_cards"])
    modules = "".join(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>' for h, p in tr["modules"])
    pipes = "".join(
        f'<div class="pipe"><div class="num">{i}</div><h3>{node}</h3><p>{detail}</p></div>'
        for i, (node, detail) in enumerate(zip(tr["diagram_nodes"], tr["diagram_details"]), 1)
    )
    services = "".join(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>' for h, p in tr["services"])
    steps = "".join(f'<div class="step"><strong>{h}</strong><p>{p}</p></div>' for h, p in tr["pilot_steps"])
    deliverables = "".join(f'<li>{x}</li>' for x in tr["deliverables"])
    yes = "".join(f'<li>{x}</li>' for x in tr["scope_yes"])
    conditions = "".join(f'<li>{x}</li>' for x in tr["scope_conditions"])
    return f"""<!DOCTYPE html>
<html lang="{tr['lang']}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{tr['title']}</title>
  <style>{base_css(False)}</style>
</head>
<body>
  <header class="nav"><div class="brand">Industrial AI MVP</div><nav>{nav}</nav></header>
  <main>
    <section class="section hero" id="s1">
      <div>
        <div class="badges">{badges}</div>
        <h1>{tr['hero_title']}</h1>
        <p class="lead">{tr['hero_subtitle']}</p>
        <p class="bodycopy">{tr['hero_body']}</p>
        <div class="actions"><a class="btn primary" href="#s2">{tr['cta_primary']}</a><a class="btn" href="#s7">{tr['cta_secondary']}</a></div>
      </div>
    </section>
    <section class="section" id="s2">
      <h2>{tr['problem_title']}</h2><p class="lead">{tr['problem_subtitle']}</p>
      <div class="grid">{problem}</div>
    </section>
    <section class="section" id="s3">
      <h2>{tr['product_title']}</h2><p class="lead">{tr['product_subtitle']}</p>
      <div class="grid">{modules}</div>
    </section>
    <section class="section">
      <h2>{tr['capabilities_title']}</h2><p class="lead">{tr['capabilities_subtitle']}</p>
      <div class="grid three">{render_capabilities(tr)}</div>
    </section>
    <section class="section">
      <h2>{tr['model_stack_title']}</h2>
      <div class="model-map">{render_model_stack(tr)}</div>
    </section>
    <section class="section">
      <h2>{tr['diagram_title']}</h2>
      <div class="pipeline">{pipes}</div>
    </section>
    <section class="section" id="s4">
      <h2>{tr['evidence_title']}</h2><p class="lead">{tr['evidence_subtitle']}</p>
      <div class="grid">{render_result_cards(tr)}</div>
      <div class="card" style="margin-top:18px"><h3>{tr['business_reading']}</h3><p>{tr['reading_text']}</p></div>
    </section>
    <section class="section">
      <h2>{tr['machine_title']}</h2><p class="lead">{tr['machine_subtitle']}</p>
      <div class="value-grid">{render_machine_kpis(tr)}</div>
    </section>
    <section class="section">
      <h2>{tr['services_title']}</h2>
      <div class="grid">{services}</div>
    </section>
    <section class="section" id="s5">
      <h2>{tr['contacts_title']}</h2><p class="lead">{tr['contacts_subtitle']}</p>
      <div class="contacts">{render_contacts(lang)}</div>
    </section>
    <section class="section" id="s7">
      <h2>{tr['pilot_title']}</h2>
      <div class="timeline">{steps}</div>
      <div class="split" style="margin-top:20px">
        <div class="card"><h3>{tr['deliverables_title']}</h3><ul class="clean">{deliverables}</ul></div>
        <div class="card"><h3>{tr['scope_title']}</h3><p>{tr['next_step']}</p></div>
      </div>
    </section>
    <section class="section">
      <h2>{tr['scope_title']}</h2>
      <div class="split">
        <div class="card"><h3>{tr['scope_left']}</h3><ul class="clean">{yes}</ul></div>
        <div class="card"><h3>{tr['scope_right']}</h3><ul class="clean">{conditions}</ul></div>
      </div>
    </section>
  </main>
  <footer>{tr['footer']}</footer>
</body>
</html>
"""


def slide(content: str) -> str:
    return f'<section class="slide"><div class="slide-inner">{content}</div></section>'


def render_deck(lang: str) -> str:
    tr = T[lang]
    badges = "".join(f'<span class="badge">{b}</span>' for b in tr["hero_badges"])
    problem = "".join(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>' for h, p in tr["problem_cards"])
    modules = "".join(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>' for h, p in tr["modules"])
    pipes = "".join(
        f'<div class="pipe"><div class="num">{i}</div><h3>{node}</h3><p>{detail}</p></div>'
        for i, (node, detail) in enumerate(zip(tr["diagram_nodes"], tr["diagram_details"]), 1)
    )
    services = "".join(f'<div class="card"><h3>{h}</h3><p>{p}</p></div>' for h, p in tr["services"])
    steps = "".join(f'<div class="step"><strong>{h}</strong><p>{p}</p></div>' for h, p in tr["pilot_steps"])
    deliverables = "".join(f'<li>{x}</li>' for x in tr["deliverables"])
    yes = "".join(f'<li>{x}</li>' for x in tr["scope_yes"])
    conditions = "".join(f'<li>{x}</li>' for x in tr["scope_conditions"])
    slides = [
        slide(f'<div class="badges">{badges}</div><h1>{tr["meeting_title"]}</h1><p class="lead">{tr["meeting_subtitle"]}</p><p class="bodycopy">{tr["hero_body"]}</p>'),
        slide(f'<h2>{tr["problem_title"]}</h2><p class="lead">{tr["problem_subtitle"]}</p><div class="grid">{problem}</div>'),
        slide(f'<h2>{tr["product_title"]}</h2><p class="lead">{tr["product_subtitle"]}</p><div class="grid">{modules}</div>'),
        slide(f'<h2>{tr["capabilities_title"]}</h2><p class="lead">{tr["capabilities_subtitle"]}</p><div class="grid three">{render_capabilities(tr)}</div>'),
        slide(f'<h2>{tr["model_stack_title"]}</h2><div class="model-map">{render_model_stack(tr)}</div>'),
        slide(f'<h2>{tr["diagram_title"]}</h2><div class="pipeline">{pipes}</div>'),
        slide(f'<h2>{tr["evidence_title"]}</h2><p class="lead">{tr["evidence_subtitle"]}</p><div class="grid two"><div class="card bars">{render_bars(tr)}</div><div class="card"><h3>{tr["business_reading"]}</h3><p>{tr["reading_text"]}</p></div></div>'),
        slide(f'<h2>{tr["machine_title"]}</h2><p class="lead">{tr["machine_subtitle"]}</p><div class="value-grid">{render_machine_kpis(tr)}</div>'),
        slide(f'<h2>{tr["services_title"]}</h2><div class="grid">{services}</div>'),
        slide(f'<h2>{tr["contacts_title"]}</h2><p class="lead">{tr["contacts_subtitle"]}</p><div class="contacts">{render_contacts(lang)}</div>'),
        slide(f'<h2>{tr["pilot_title"]}</h2><div class="timeline">{steps}</div>'),
        slide(f'<h2>{tr["deliverables_title"]}</h2><div class="split"><div class="card"><ul class="clean">{deliverables}</ul></div><div class="card"><h3>{tr["scope_title"]}</h3><p>{tr["next_step"]}</p></div></div>'),
        slide(f'<h2>{tr["scope_title"]}</h2><div class="split"><div class="card"><h3>{tr["scope_left"]}</h3><ul class="clean">{yes}</ul></div><div class="card"><h3>{tr["scope_right"]}</h3><ul class="clean">{conditions}</ul></div></div>'),
    ]
    slides[0] = slides[0].replace('class="slide"', 'class="slide active"', 1)
    return f"""<!DOCTYPE html>
<html lang="{tr['lang']}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{tr['title']} - Meeting Deck</title>
  <style>{base_css(True)}</style>
</head>
<body>
  <header class="nav"><div class="brand">Industrial AI MVP</div><div class="caption">{tr['pilot_title']}</div></header>
  <main class="deck">{''.join(slides)}</main>
  <div class="controls"><span id="count">1 / {len(slides)}</span><div class="dots" id="dots"></div><div><button id="prev">←</button><button id="next">→</button></div></div>
  <script>
    const slides = [...document.querySelectorAll('.slide')];
    const dots = document.getElementById('dots');
    const count = document.getElementById('count');
    let idx = 0;
    slides.forEach((_, i) => {{
      const d = document.createElement('span'); d.className = 'dot' + (i === 0 ? ' active' : '');
      d.onclick = () => go(i); dots.appendChild(d);
    }});
    function go(i) {{
      idx = Math.max(0, Math.min(slides.length - 1, i));
      slides.forEach((s, n) => s.classList.toggle('active', n === idx));
      [...document.querySelectorAll('.dot')].forEach((d, n) => d.classList.toggle('active', n === idx));
      count.textContent = (idx + 1) + ' / ' + slides.length;
    }}
    document.getElementById('next').onclick = () => go(idx + 1);
    document.getElementById('prev').onclick = () => go(idx - 1);
    document.addEventListener('keydown', e => {{ if (e.key === 'ArrowRight' || e.key === ' ') go(idx + 1); if (e.key === 'ArrowLeft') go(idx - 1); }});
  </script>
</body>
</html>
"""


def find_browser() -> Path | None:
    for path in EDGE_CANDIDATES:
        if path.exists():
            return path
    return None


def html_to_pdf(browser: Path, html: Path, pdf: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    url = html.resolve().as_uri()
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf.resolve()}",
        url,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build commercial HTML/PDF presentations.")
    parser.add_argument("--out-dir", default="presentation")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "mvp_galicia.html": render_web("es"),
        "mvp_galicia_reunion.html": render_deck("es"),
        "mvp_galicia_en.html": render_web("en"),
        "mvp_galicia_reunion_en.html": render_deck("en"),
    }
    for name, html in files.items():
        (out / name).write_text(html, encoding="utf-8")
        print(f"wrote {out / name}")

    if args.pdf:
        browser = find_browser()
        if browser is None:
            raise SystemExit("No Edge/Chrome executable found for PDF export.")
        for name in files:
            html_to_pdf(browser, out / name, out / name.replace(".html", ".pdf"))
            print(f"wrote {out / name.replace('.html', '.pdf')}")


if __name__ == "__main__":
    main()
