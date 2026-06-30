# Claims and Limitations

| Claim | Estado | Evidencia necesaria | Riesgo | Cómo validarlo |
|---|---|---|---|---|
| MVP técnico para anomaly detection industrial | permitido | scripts, tests, manifest, visual benchmark, demo | dataset público no equivale a planta | piloto con datos reales |
| Pipeline de validación con baselines fuertes | permitido | PatchCore/PaDiM/pixel baseline y tablas | fallback no preentrenado si DINOv2 no está local | registrar backbone real usado |
| DINOv2/PatchCore/PaDiM como baseline visual fuerte | permitido en quick run | DINOv2 real cargado desde caché local y benchmark MVTec bottle quick | extrapolar a todas las categorías | ejecutar benchmark multi-categoría |
| DINOv3 como sustitución futura | parcial | requeriría pesos locales y benchmark repetido | confundir roadmap con resultado | mantener `actual_backbone` en reportes |
| LeJEPA in-domain es experimental | permitido | SIGReg, logs, collapse diagnostics | sobreventa sin delta downstream | frozen probes y anomaly benchmark |
| LeWorldModel industrial es experimental | permitido | prediction loss y surprise benchmark | pseudo-temporal sin acciones reales | usar setpoints reales y split temporal |
| Predictive quality requiere datos secuenciales o acciones/setpoints | permitido | manifest con temporalidad/actions | usar metadata estática como acción causal | etiquetar contexto vs acción real |
| Control autónomo industrial | prohibido | requeriría integración, seguridad y validación | riesgo operacional/legal | no usar este claim |
| Causalidad | prohibido | requeriría diseño causal/intervenciones | confundir correlación con acción | usar wording de correlación y sorpresa |
| Listo para producción | prohibido | falta integración, monitorización, seguridad, drift | expectativa comercial incorrecta | posicionar como piloto |
| Supera soluciones industriales comerciales | prohibido | requeriría comparativa formal | no hay evidencia | no usar este claim |
