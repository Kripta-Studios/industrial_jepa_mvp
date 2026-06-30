ESTADO REAL DE LOS DATASETS DESCARGADOS

Sensor-JEPA:
  CNC Milling
  CWRU Bearing
  Paderborn Bearing

Visual-JEPA:
  MVTec AD
  VisA
  KolektorSDD

Están en "C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\industrial_jepa_mvp\data\raw"


Quiero implementar desde cero un MVP comercial/técnico basado en JEPA para dos líneas de producto:

A. Mantenimiento predictivo / maquinaria / sensores industriales
B. Visión industrial / defectos / control de calidad

El objetivo NO es hacer investigación académica infinita. El objetivo es construir dos demos vendibles, reproducibles y defendibles para enseñar a empresas industriales que un modelo auto-supervisado tipo JEPA puede aprender representaciones útiles a partir de datos no etiquetados y luego resolver tareas downstream con pocas etiquetas.

Contexto conceptual:

JEPA = Joint-Embedding Predictive Architecture. La idea es no reconstruir señales brutas, píxeles o imágenes completas, sino aprender a predecir representaciones latentes de partes ocultas usando contexto visible.

Para sensores industriales:

* tomar ventanas temporales multicanal;
* ocultar spans temporales, canales o bloques;
* codificar contexto visible;
* codificar target oculto;
* predecir embedding latente del target;
* usar el encoder para desgaste, RUL, anomalía o clasificación de fallo.

Para visión industrial:

* tomar imágenes o patches industriales;
* ocultar regiones, patches o vistas;
* codificar contexto;
* predecir embedding de regiones target;
* usar el encoder para anomalía, segmentación de defectos, clasificación normal/defectuoso y few-shot defect detection.

Quiero un repositorio limpio, modular y reproducible.

────────────────────────────────────────
ESTRUCTURA GENERAL DEL REPO
────────────────────────────────────────

Crear esta estructura:

industrial_jepa_mvp/
README.md
README_DATASETS.md
README_EXPERIMENTS.md
requirements.txt
pyproject.toml

configs/
sensor_jepa/
cnc_milling.yaml
multi_sensor_cnc.yaml
paderborn_bearing.yaml
cwru_bearing.yaml
nasa_ims_bearing.yaml
cmapss.yaml
ai4i.yaml
demo_sensor_quick.yaml

```
visual_jepa/
  mvtec_ad.yaml
  mvtec_3d_ad.yaml
  visa.yaml
  kolektor_sdd.yaml
  neu_surface.yaml
  dagm_2007.yaml
  severstal.yaml
  demo_visual_quick.yaml
```

data/
raw/
processed/
interim/
manifests/

src/
common/
seed.py
logging.py
paths.py
metrics.py
reports.py
plots.py
config.py

```
sensor_jepa/
  data/
    base.py
    windowing.py
    normalization.py
    cnc_milling.py
    multi_sensor_cnc.py
    paderborn.py
    cwru.py
    nasa_ims.py
    cmapss.py
    ai4i.py
  models/
    encoders.py
    predictors.py
    heads.py
    losses.py
    sensor_jepa.py
    baselines.py
  train/
    pretrain.py
    probe.py
    finetune.py
    evaluate.py
  eval/
    metrics.py
    rul_metrics.py
    anomaly_metrics.py
    visualization.py

visual_jepa/
  data/
    base.py
    transforms.py
    patching.py
    mvtec_ad.py
    mvtec_3d_ad.py
    visa.py
    kolektor_sdd.py
    neu_surface.py
    dagm.py
    severstal.py
  models/
    encoders.py
    predictors.py
    heads.py
    losses.py
    visual_jepa.py
    baselines.py
  train/
    pretrain.py
    probe.py
    finetune.py
    evaluate.py
  eval/
    metrics.py
    anomaly_maps.py
    visualization.py
```

scripts/
00_create_dataset_manifest.py
01_download_or_prepare_datasets.py
02_prepare_sensor_data.py
03_prepare_visual_data.py
04_pretrain_sensor_jepa.py
05_probe_sensor_jepa.py
06_finetune_sensor_jepa.py
07_eval_sensor_jepa.py
08_pretrain_visual_jepa.py
09_probe_visual_jepa.py
10_finetune_visual_jepa.py
11_eval_visual_jepa.py
12_compare_experiments.py
13_export_sensor_demo_report.py
14_export_visual_demo_report.py
15_run_sensor_demo.py
16_run_visual_demo.py
17_run_all_demos.py

outputs/
sensor_jepa/
visual_jepa/
reports/
demo_artifacts/

notebooks/
sensor_exploration.ipynb
visual_exploration.ipynb

tests/
test_sensor_windowing.py
test_visual_patching.py
test_freeze_encoder.py
test_metrics.py
test_dataset_manifest.py

────────────────────────────────────────
OBJETIVO DE PRODUCTO
────────────────────────────────────────

Construir dos MVPs:

MVP A — Sensor-JEPA Predictive Maintenance

Producto demo:

“Sensor-JEPA aprende representaciones de señales industriales no etiquetadas para detectar desgaste, predecir vida útil restante, clasificar estado de máquina y detectar anomalías.”

Casos de uso comerciales:

* máquinas CNC;
* rodamientos;
* motores;
* líneas de fabricación;
* vibración/corriente;
* sensores SCADA;
* mantenimiento predictivo;
* detección temprana de fallo;
* reducción de etiquetado manual.

Empresas objetivo:

* Navantia;
* Televés;
* FINSA;
* Norvento;
* Repsol;
* Hijos de Rivera;
* Jealsa;
* Frinsa;
* fábricas pequeñas con sensores de máquina.

MVP B — Visual-JEPA Industrial Quality

Producto demo:

“Visual-JEPA aprende representaciones visuales de productos normales y localiza defectos/anomalías con pocas etiquetas o incluso sin ejemplos defectuosos durante entrenamiento.”

Casos de uso comerciales:

* control de calidad visual;
* defectos superficiales;
* textura;
* piezas industriales;
* madera;
* metal;
* placas electrónicas;
* envases;
* latas;
* botellas;
* empaquetado;
* inspección automática.

Empresas objetivo:

* FINSA;
* Televés;
* Jealsa;
* Frinsa;
* Hijos de Rivera;
* Nueva Pescanova;
* Navantia;
* Cinfo como posible integrador visual.

────────────────────────────────────────
DATASETS A DESCARGAR — PROYECTO A
MANTENIMIENTO PREDICTIVO / SENSORES
────────────────────────────────────────

Implementar soporte para estos datasets, en este orden de prioridad.

A1. CNC Milling Process Dataset

Prioridad: CRÍTICA.

Uso:

* demo principal de desgaste de herramienta;
* señales de vibración;
* señales de corriente;
* ciclos de fresado;
* tool life;
* clasificación de condición;
* regresión de vida útil o desgaste;
* representación multicanal.

Datos esperados:

* archivos raw tipo Pxxx_Fyy_Czz.csv;
* FeatureAndMetadata_Milling.csv;
* metadata por ciclo;
* tool number;
* sample number;
* hardness;
* señales de vibración;
* señales de corriente.

Tareas downstream:

* clasificación de estado de herramienta;
* estimación de vida útil;
* predicción de ciclo avanzado;
* detección de ciclo anómalo;
* clustering de ciclos similares.

Implementar loader:
src/sensor_jepa/data/cnc_milling.py

Debe:

* leer features procesadas si están disponibles;
* leer raw CSV si están disponibles;
* mapear cada ciclo a tool_id;
* mapear sample/cycle_id;
* crear ventanas temporales multicanal;
* normalizar por canal;
* permitir split por herramienta para evitar leakage;
* guardar processed parquet.

Splits recomendados:

* split_by_tool: algunas herramientas train, otras val/test;
* split_by_life_stage: early/mid/late;
* split_by_cycle: solo para debug, no como evaluación principal.

Métricas:

* classification: accuracy, macro-F1, balanced accuracy, AUROC si binario;
* regression: MAE, RMSE, R2;
* RUL: MAE, RMSE, NASA score opcional;
* anomaly: AUROC, AUPRC, precision@k.

A2. Multi-Sensor CNC Tool Wear Dataset

Prioridad: ALTA.

Uso:

* demo rápida y limpia;
* fuerza Fx/Fy/Fz;
* vibración X/Y/Z;
* Acoustic Emission;
* VB_mm;
* Wear_Class: Healthy, Moderate, Worn.

Implementar loader:
src/sensor_jepa/data/multi_sensor_cnc.py

Debe:

* detectar columnas de sensores;
* detectar target VB_mm;
* detectar Wear_Class;
* permitir classification y regression;
* crear ventanas temporales;
* normalizar por sensor;
* soportar entrenamiento rápido.

Tareas:

* Wear_Class classification;
* VB_mm regression;
* anomaly detection usando Healthy como normal;
* few-label classification: entrenar con 5%, 10%, 25%, 100% de etiquetas.

A3. Paderborn Bearing Dataset

Prioridad: ALTA.

Uso:

* condición de rodamientos;
* vibración + corriente;
* fallos reales/artificiales;
* condition monitoring.

Implementar loader:
src/sensor_jepa/data/paderborn.py

Debe:

* permitir path manual al dataset descargado;
* leer señales de vibración y corriente;
* extraer operating condition;
* extraer bearing state;
* mapear healthy/damaged;
* mapear tipo de fallo si existe;
* crear ventanas.

Tareas:

* healthy vs damaged;
* tipo de daño;
* dominio por operating condition;
* pretrain en una condición y transfer a otra;
* frozen probe para evaluar representación.

A4. CWRU Bearing Dataset

Prioridad: ALTA.

Uso:

* benchmark clásico;
* fácil de explicar;
* vibración;
* defectos en drive end / fan end;
* distintas cargas.

Implementar loader:
src/sensor_jepa/data/cwru.py

Debe:

* leer archivos .mat;
* extraer señales drive end, fan end, RPM;
* mapear defect type;
* mapear defect size;
* mapear load;
* crear ventanas;
* soportar sampling rate 12k y 48k.

Tareas:

* normal vs fault;
* fault type classification;
* transfer across load;
* few-label classification.

A5. NASA IMS Bearing Dataset

Prioridad: MEDIA-ALTA.

Uso:

* run-to-failure;
* bearing degradation;
* anomalía temporal;
* early warning.

Implementar loader:
src/sensor_jepa/data/nasa_ims.py

Debe:

* leer snapshots temporales;
* ordenar por timestamp;
* mapear bearing_id;
* construir serie de degradación;
* generar pseudo-RUL si no viene explícito;
* crear ventanas por bearing.

Tareas:

* anomaly detection;
* degradation stage classification;
* RUL proxy;
* lead-time detection.

A6. NASA C-MAPSS Turbofan Engine Dataset

Prioridad: ALTA.

Uso:

* RUL multivariante;
* series por motor;
* benchmark muy conocido;
* útil para explicar vida útil restante.

Implementar loader:
src/sensor_jepa/data/cmapss.py

Debe:

* leer FD001, FD002, FD003, FD004;
* identificar unit_id, cycle, settings, sensors;
* calcular RUL;
* aplicar clipping opcional de RUL;
* crear ventanas de longitud configurable;
* split train/test oficial.

Tareas:

* RUL regression;
* health stage classification;
* anomaly score por motor;
* frozen encoder + regression head.

A7. AI4I 2020 Predictive Maintenance Dataset

Prioridad: MEDIA.

Uso:

* dataset pequeño y sintético;
* demo tabular rápida;
* no debe ser la demo principal;
* útil para UI/dashboard y explicación comercial.

Implementar loader:
src/sensor_jepa/data/ai4i.py

Debe:

* leer CSV;
* detectar Machine failure;
* detectar failure modes: TWF, HDF, PWF, OSF, RNF;
* normalizar variables;
* soportar tabular encoder simple.

Tareas:

* binary machine failure;
* failure mode multilabel;
* baseline LightGBM/XGBoost;
* comparación con Tabular-JEPA opcional.

A8. Tennessee Eastman Process Dataset

Prioridad: OPCIONAL / EXTENSIÓN.

Uso:

* procesos químicos;
* fault detection;
* útil para Repsol/Zendal/Hijos de Rivera como analogía de proceso industrial.

Implementar solo si lo anterior ya funciona.

Tareas:

* normal vs fault;
* fault type classification;
* anomaly detection;
* transfer between operating modes.

────────────────────────────────────────
DATASETS A DESCARGAR — PROYECTO B
VISIÓN INDUSTRIAL / DEFECTOS / CONTROL DE CALIDAD
────────────────────────────────────────

Implementar soporte para estos datasets, en este orden de prioridad.

B1. MVTec AD

Prioridad: CRÍTICA.

Uso:

* demo principal de anomalía visual industrial;
* entrenamiento con imágenes normales;
* test con normales + defectuosas;
* máscaras de anomalía;
* 15 categorías de objetos/texturas.

Implementar loader:
src/visual_jepa/data/mvtec_ad.py

Debe:

* leer estructura por categoría;
* train = normal;
* test = normal + anomaly;
* ground_truth masks si existen;
* soportar category específica o all_categories;
* redimensionar imágenes;
* generar patches;
* normalizar;
* crear manifest parquet/csv.

Tareas:

* anomaly classification image-level;
* anomaly localization pixel-level;
* few-shot defect classification;
* pretrain en normales;
* frozen probe.

Categorías prioritarias para demo:

* bottle;
* cable;
* capsule;
* metal_nut;
* screw;
* tile;
* wood;
* transistor;
* zipper;
* hazelnut.

Métricas:

* image AUROC;
* pixel AUROC;
* AUPRC;
* PRO score si se implementa;
* IoU/F1 con threshold;
* false positive rate.

B2. MVTec 3D-AD

Prioridad: ALTA.

Uso:

* demo RGB + 3D;
* útil para piezas industriales;
* útil para conectar con LiDAR/3D;
* defectos geométricos.

Implementar loader:
src/visual_jepa/data/mvtec_3d_ad.py

Debe:

* leer RGB;
* leer depth/point cloud si existe;
* leer masks;
* permitir modalidad rgb_only, depth_only, rgb_depth;
* crear patches multimodales.

Tareas:

* anomaly detection RGB;
* anomaly detection depth;
* multimodal RGB+depth;
* JEPA cross-modal:

  * RGB context predice depth latent target;
  * depth context predice RGB latent target;
  * RGB+depth context predice target patch latent.

B3. VisA Industrial Anomaly Dataset

Prioridad: ALTA.

Uso:

* dataset grande de anomalía visual;
* 12 clases;
* PCBs y objetos;
* bueno para Televés por PCB/electrónica;
* bueno para generalización.

Implementar loader:
src/visual_jepa/data/visa.py

Debe:

* leer estructura de clases;
* separar normal/anomaly;
* leer masks/anotaciones si existen;
* soportar categorías PCB;
* crear split train/val/test reproducible.

Tareas:

* anomaly classification;
* anomaly localization;
* transfer MVTec -> VisA;
* few-shot defect adaptation.

B4. Kolektor Surface-Defect Dataset / KolektorSDD

Prioridad: ALTA.

Uso:

* defectos reales de superficie industrial;
* dataset más cercano a línea de fabricación;
* útil para demo con metal/plástico/superficie.

Implementar loader:
src/visual_jepa/data/kolektor_sdd.py

Debe:

* leer imágenes;
* leer anotaciones/masks;
* soportar folds si existen;
* generar train/val/test.

Tareas:

* binary defect detection;
* segmentation defect/no defect;
* few-shot training;
* comparison with MVTec pretraining.

B5. NEU Surface Defect / NEU-DET

Prioridad: MEDIA-ALTA.

Uso:

* defectos de acero;
* útil para Navantia, metal, fabricación;
* clasificación de 6 tipos de defectos;
* detección/segmentación si se usa NEU-DET.

Implementar loader:
src/visual_jepa/data/neu_surface.py

Debe:

* leer imágenes grayscale;
* mapear clases:

  * crazing;
  * inclusion;
  * patches;
  * pitted_surface;
  * rolled-in_scale;
  * scratches;
* soportar clasificación;
* si hay bounding boxes, soportar detección/segmentación débil.

Tareas:

* defect type classification;
* defect vs normal si se construye normal;
* embeddings de defectos;
* few-shot classification.

B6. DAGM 2007

Prioridad: MEDIA.

Uso:

* texturas industriales sintéticas;
* defectos sutiles;
* buen test para anomalía en texturas;
* no vender como dataset real, sino como benchmark.

Implementar loader:
src/visual_jepa/data/dagm.py

Debe:

* leer clases;
* leer imágenes non-defective/defective;
* leer masks;
* soportar fold train/test.

Tareas:

* anomaly localization;
* texture defect detection;
* pretraining en normales.

B7. Severstal Steel Defect Detection

Prioridad: MEDIA.

Uso:

* segmentación de defectos en acero;
* dataset grande Kaggle;
* útil para demo metal/naval/industrial.

Implementar loader:
src/visual_jepa/data/severstal.py

Debe:

* leer imágenes;
* leer run-length encoded masks;
* convertir RLE a máscara binaria/multiclase;
* soportar 4 clases de defectos;
* crear train/val split.

Tareas:

* segmentation supervised;
* JEPA pretraining + segmentation head;
* defect/no defect;
* few-label segmentation.

B8. Wood Surface Defect Dataset

Prioridad: OPCIONAL / ESPECIAL PARA FINSA.

Uso:

* demo para madera;
* útil si se quiere visitar FINSA;
* defectos tipo nudos, grietas, resina, etc.

Implementar solo si lo anterior ya funciona.

Tareas:

* defect classification;
* anomaly localization;
* wood-specific report.

────────────────────────────────────────
ARQUITECTURA MVP A — SENSOR-JEPA
────────────────────────────────────────

Implementar modelo mínimo pero serio.

Input:

* batch de ventanas temporales;
* shape: [B, T, C] o [B, C, T];
* T = longitud temporal;
* C = canales/sensores.

Preprocesamiento:

* resampling opcional;
* normalización por canal;
* clipping/winsorization opcional;
* features derivadas opcionales:

  * RMS;
  * kurtosis;
  * skewness;
  * spectral centroid;
  * band power;
  * FFT summary;
  * pero no depender solo de features manuales.

Masking:

* temporal span masking;
* channel masking;
* block masking temporal+canal;
* future target prediction opcional;
* target windows separadas del contexto.

Encoder:

* implementar al menos dos:

  1. Conv1DEncoder:

     * Conv1D residual;
     * pooling temporal;
     * embedding_dim configurable.
  2. TransformerTimeSeriesEncoder:

     * patchify temporal;
     * positional encoding;
     * Transformer encoder;
     * embedding global o embeddings por patch.

Predictor:

* MLP predictor;
* Transformer predictor opcional;
* input: context embedding + target positional/metadata embedding;
* output: predicted target embedding.

Target encoder:

* modo shared:

  * mismo encoder para context y target;
  * sin EMA.
* modo ema opcional:

  * solo si ya está implementado fácilmente.
* por defecto usar shared para estilo LeJEPA simple.

Loss:

* cosine loss o MSE latente entre pred_target_emb y target_emb;
* SIGReg o regularización anti-colapso:

  * variance regularization;
  * covariance regularization opcional;
  * embedding_std logging;
* guardar métricas de colapso:

  * embedding mean;
  * embedding std;
  * cosine similarity distribution.

Downstream heads:

* classification head;
* regression head;
* anomaly head;
* RUL head.

Modos de evaluación:

* frozen linear probe;
* frozen MLP probe;
* semi-frozen encoder;
* full fine-tune.

Flags obligatorios:

* --freeze-encoder
* --encoder-lr-scale
* --probe-type linear|mlp
* --task classification|regression|rul|anomaly
* --label-fraction 0.01|0.05|0.10|0.25|1.0

Baselines:

* XGBoost/LightGBM si están disponibles;
* RandomForest;
* 1D CNN supervised;
* LSTM/GRU supervised;
* Autoencoder anomaly;
* IsolationForest sobre features;
* PCA + classifier.

No bloquear el MVP si no está LightGBM instalado. Hacer fallback a sklearn.

────────────────────────────────────────
ARQUITECTURA MVP B — VISUAL-JEPA
────────────────────────────────────────

Input:

* imágenes RGB o grayscale;
* opcional depth/3D para MVTec 3D-AD;
* shape [B, C, H, W].

Preprocesamiento:

* resize configurable, por ejemplo 224x224;
* normalización ImageNet opcional;
* augmentations suaves:

  * random crop;
  * horizontal flip;
  * color jitter leve;
  * no deformar defectos en evaluación.

Patchification:

* dividir imagen en patches;
* mask ratio configurable;
* context patches visibles;
* target patches ocultos;
* soportar multi-block masking.

Encoder:

* implementar al menos:

  1. SmallConvEncoder para quick demo.
  2. ViT-like encoder pequeño si el tiempo lo permite.
* Permitir usar backbone pretrained opcional:

  * ResNet18;
  * ViT small;
  * pero el MVP debe funcionar sin depender de modelos enormes.

Predictor:

* MLP predictor;
* Transformer predictor opcional;
* predice embeddings target desde embeddings context.

Loss:

* cosine/MSE latent;
* variance regularization/SIGReg;
* logging de embedding_std;
* no reconstrucción de píxeles como objetivo principal.

Anomaly scoring:

* método 1:

  * distancia entre embedding predicho y embedding target;
  * anomalía por patch.
* método 2:

  * memoria de embeddings normales tipo PatchCore simple;
  * kNN distance en espacio latente.
* método 3:

  * classifier head para normal/defective si hay etiquetas.

Outputs visuales:

* anomaly heatmap;
* overlay sobre imagen;
* mask predicha;
* comparación con ground truth;
* top-k anomalías.

Métricas:

* image-level AUROC;
* pixel-level AUROC;
* AUPRC;
* IoU/F1 tras threshold;
* precision@k;
* false positive rate;
* per-category metrics.

Modos:

* train only normal images;
* train normal + few anomalies;
* frozen linear probe;
* frozen MLP probe;
* semi-frozen;
* full fine-tune;
* transfer MVTec → VisA o MVTec → Kolektor.

Baselines:

* Autoencoder;
* PaDiM simple si es posible;
* PatchCore simple si es posible;
* ResNet features + kNN;
* supervised ResNet classifier;
* UNet segmentation para datasets con masks si es posible.

────────────────────────────────────────
EXPERIMENTOS OBLIGATORIOS — SENSOR-JEPA
────────────────────────────────────────

Crear script:

scripts/15_run_sensor_demo.py

Debe ejecutar, mínimo, sobre CNC Milling o Multi-Sensor CNC:

1. Preparar dataset.
2. Entrenar baseline supervised.
3. Pretrain Sensor-JEPA con datos no etiquetados.
4. Frozen linear probe.
5. Frozen MLP probe.
6. Semi-frozen fine-tune.
7. Full fine-tune.
8. Evaluar.
9. Comparar.
10. Exportar report.

Outputs:

outputs/sensor_jepa/<dataset_name>/
baseline/
pretrain/
frozen_linear/
frozen_mlp/
semifrozen/
full_finetune/
comparison/
test_comparison.csv
test_comparison.md
metrics_summary.json
plots/
confusion_matrix.png
embedding_umap.png
wear_over_life.png
rul_prediction.png
anomaly_scores.png
report/
sensor_jepa_demo_report.md
sensor_jepa_demo_report.html

Métricas por tarea:

Classification:

* accuracy;
* balanced accuracy;
* macro-F1;
* per-class F1;
* confusion matrix.

Regression / wear / VB_mm:

* MAE;
* RMSE;
* R2;
* Spearman correlation;
* prediction vs true plot.

RUL:

* MAE;
* RMSE;
* NASA score opcional;
* early/late error;
* RUL curve per unit/tool.

Anomaly:

* AUROC;
* AUPRC;
* precision@k;
* detection lead time si hay run-to-failure;
* false alarm rate.

Comparación mínima:

model_name, encoder_mode, probe_type, label_fraction, task, accuracy, balanced_accuracy, macro_F1, MAE, RMSE, AUROC, AUPRC, trainable_params, frozen_encoder, encoder_lr_scale

Criterio de éxito Sensor-JEPA:

* frozen_linear > baseline simple en al menos una métrica clave, o
* frozen_mlp competitivo con baseline usando menos etiquetas, o
* semifrozen/full_finetune mejora baseline con estabilidad, o
* anomaly score detecta degradación antes del fallo.

Conclusión automática:

Si frozen probe funciona:
“Sensor-JEPA aprende representaciones industriales reutilizables.”

Si solo full fine-tune funciona:
“Sensor-JEPA ayuda como inicialización, pero falta demostrar representación linealmente separable.”

Si no mejora:
“Hay que revisar masking, normalización, split por máquina/herramienta o backbone.”

────────────────────────────────────────
EXPERIMENTOS OBLIGATORIOS — VISUAL-JEPA
────────────────────────────────────────

Crear script:

scripts/16_run_visual_demo.py

Debe ejecutar, mínimo, sobre MVTec AD con una categoría inicial:

* bottle o metal_nut para objeto;
* tile o wood para textura;
* transistor para electrónica.

Pipeline:

1. Preparar dataset.
2. Entrenar baseline anomaly detector.
3. Pretrain Visual-JEPA con imágenes normales.
4. Evaluar anomaly scoring con latent prediction error.
5. Entrenar frozen linear probe con pocas etiquetas.
6. Entrenar frozen MLP probe.
7. Semi-frozen.
8. Full fine-tune si hay tarea supervisada.
9. Exportar heatmaps.
10. Exportar report.

Outputs:

outputs/visual_jepa/<dataset_name>/<category>/
baseline/
pretrain/
frozen_linear/
frozen_mlp/
semifrozen/
full_finetune/
anomaly_maps/
sample_001_input.png
sample_001_gt_mask.png
sample_001_pred_heatmap.png
sample_001_overlay.png
sample_001_thresholded_mask.png
comparison/
test_comparison.csv
test_comparison.md
per_category_metrics.csv
report/
visual_jepa_demo_report.md
visual_jepa_demo_report.html

Métricas:

Image-level:

* AUROC;
* AUPRC;
* accuracy con threshold;
* F1;
* false positive rate.

Pixel-level:

* AUROC;
* AUPRC;
* IoU;
* Dice/F1;
* PRO score opcional.

Few-shot:

* label_fraction;
* macro-F1;
* balanced accuracy.

Comparación mínima:

model_name, category, train_mode, anomaly_method, image_AUROC, image_AUPRC, pixel_AUROC, pixel_AUPRC, pixel_IoU, pixel_F1, threshold, trainable_params, frozen_encoder

Criterio de éxito Visual-JEPA:

* anomaly heatmaps localizan defectos razonablemente;
* image-level AUROC competitivo frente a baseline simple;
* frozen/few-shot probe mejora con pocas etiquetas;
* Visual-JEPA produce embeddings útiles para normal/defectuoso.

Conclusión automática:

Si JEPA mejora heatmaps o few-shot:
“Visual-JEPA aprende representaciones visuales útiles para control de calidad.”

Si solo supervised fine-tune funciona:
“JEPA ayuda como inicialización, pero falta mejorar anomaly scoring auto-supervisado.”

Si no mejora:
“Revisar masking, patch size, backbone, normalización y scoring.”

────────────────────────────────────────
DESCARGA Y MANIFEST DE DATASETS
────────────────────────────────────────

Crear:

scripts/00_create_dataset_manifest.py

Debe generar:

data/manifests/datasets.yaml

Con esta estructura:

sensor_jepa:
cnc_milling:
name: "CNC Milling Process Dataset"
priority: "critical"
source_type: "figshare/nature/kaggle"
task_types: ["tool_wear", "classification", "regression", "rul", "anomaly"]
expected_files:
- "FeatureAndMetadata_Milling.csv"
- "Pxxx_Fyy_Czz.csv"
local_path: "data/raw/sensor/cnc_milling"

multi_sensor_cnc:
name: "Multi-Sensor CNC Tool Wear Dataset"
priority: "high"
source_type: "kaggle"
task_types: ["tool_wear", "classification", "regression"]
expected_columns:
- "Fx"
- "Fy"
- "Fz"
- "Vibration_X"
- "Vibration_Y"
- "Vibration_Z"
- "AE"
- "VB_mm"
- "Wear_Class"
local_path: "data/raw/sensor/multi_sensor_cnc"

paderborn_bearing:
name: "Paderborn Bearing Dataset"
priority: "high"
source_type: "official"
task_types: ["bearing_fault", "condition_monitoring", "domain_transfer"]
local_path: "data/raw/sensor/paderborn"

cwru_bearing:
name: "CWRU Bearing Dataset"
priority: "high"
source_type: "official"
task_types: ["bearing_fault", "classification"]
local_path: "data/raw/sensor/cwru"

nasa_ims_bearing:
name: "NASA IMS Bearing Dataset"
priority: "medium_high"
source_type: "nasa"
task_types: ["run_to_failure", "anomaly", "rul_proxy"]
local_path: "data/raw/sensor/nasa_ims"

cmapss:
name: "NASA C-MAPSS"
priority: "high"
source_type: "nasa"
task_types: ["rul", "multivariate_timeseries"]
local_path: "data/raw/sensor/cmapss"

ai4i:
name: "AI4I 2020 Predictive Maintenance"
priority: "medium"
source_type: "uci"
task_types: ["tabular", "failure_classification"]
local_path: "data/raw/sensor/ai4i"

visual_jepa:
mvtec_ad:
name: "MVTec AD"
priority: "critical"
source_type: "official"
task_types: ["anomaly_detection", "localization", "segmentation"]
local_path: "data/raw/visual/mvtec_ad"

mvtec_3d_ad:
name: "MVTec 3D-AD"
priority: "high"
source_type: "official"
task_types: ["rgb_3d_anomaly", "localization"]
local_path: "data/raw/visual/mvtec_3d_ad"

visa:
name: "VisA"
priority: "high"
source_type: "aws"
task_types: ["anomaly_detection", "localization", "fewshot"]
local_path: "data/raw/visual/visa"

kolektor_sdd:
name: "KolektorSDD"
priority: "high"
source_type: "official"
task_types: ["surface_defect", "segmentation"]
local_path: "data/raw/visual/kolektor_sdd"

neu_surface:
name: "NEU Surface Defect / NEU-DET"
priority: "medium_high"
source_type: "official/kaggle"
task_types: ["steel_defect", "classification", "detection"]
local_path: "data/raw/visual/neu_surface"

dagm_2007:
name: "DAGM 2007"
priority: "medium"
source_type: "official/kaggle"
task_types: ["texture_anomaly", "weak_supervision"]
local_path: "data/raw/visual/dagm_2007"

severstal:
name: "Severstal Steel Defect Detection"
priority: "medium"
source_type: "kaggle"
task_types: ["steel_defect", "segmentation"]
local_path: "data/raw/visual/severstal"

No descargar automáticamente datasets que requieran login, Kaggle API o aceptar licencia. En esos casos:

* imprimir instrucciones;
* comprobar si el path existe;
* validar archivos esperados;
* generar mensaje claro si falta algo.

Crear README_DATASETS.md con:

* nombre del dataset;
* para qué sirve;
* cómo descargarlo;
* dónde colocarlo;
* archivos esperados;
* tarea recomendada;
* prioridad;
* empresa/vertical asociada.

────────────────────────────────────────
README_DATASETS — CONTENIDO OBLIGATORIO
────────────────────────────────────────

Incluir tabla:

Dataset | Proyecto | Prioridad | Tipo | Tarea | Vertical comercial | Ruta local

Ejemplo:

CNC Milling Process Dataset | Sensor-JEPA | Crítica | vibración/corriente CNC | desgaste/RUL/anomalía | Navantia, Televés, FINSA, Norvento | data/raw/sensor/cnc_milling

Multi-Sensor CNC Tool Wear | Sensor-JEPA | Alta | fuerza/vibración/AE | VB_mm, Wear_Class | fabricación CNC | data/raw/sensor/multi_sensor_cnc

Paderborn Bearing | Sensor-JEPA | Alta | vibración/corriente | fallos rodamiento | fábricas, renovables, maquinaria | data/raw/sensor/paderborn

CWRU Bearing | Sensor-JEPA | Alta | vibración | fault classification | maquinaria | data/raw/sensor/cwru

NASA IMS Bearing | Sensor-JEPA | Media-alta | run-to-failure | anomalía/RUL | mantenimiento predictivo | data/raw/sensor/nasa_ims

NASA C-MAPSS | Sensor-JEPA | Alta | multivariate time series | RUL | motores/turbinas | data/raw/sensor/cmapss

AI4I 2020 | Sensor-JEPA | Media | tabular sintético | failure classification | demo rápida | data/raw/sensor/ai4i

MVTec AD | Visual-JEPA | Crítica | imágenes industriales | anomalía/localización | control calidad | data/raw/visual/mvtec_ad

MVTec 3D-AD | Visual-JEPA | Alta | RGB+3D | anomalía geométrica | piezas industriales/3D | data/raw/visual/mvtec_3d_ad

VisA | Visual-JEPA | Alta | imágenes industriales/PCB | anomalía/localización | electrónica/Televés | data/raw/visual/visa

KolektorSDD | Visual-JEPA | Alta | superficie real | defecto/segmentación | fabricación | data/raw/visual/kolektor_sdd

NEU Surface Defect | Visual-JEPA | Media-alta | acero | defect type | metal/naval | data/raw/visual/neu_surface

DAGM 2007 | Visual-JEPA | Media | textura sintética | anomalía textura | benchmark | data/raw/visual/dagm_2007

Severstal | Visual-JEPA | Media | acero | segmentación defectos | metal/naval | data/raw/visual/severstal

────────────────────────────────────────
CONFIGS MÍNIMOS
────────────────────────────────────────

Crear configs YAML reproducibles.

Ejemplo Sensor-JEPA quick:

configs/sensor_jepa/demo_sensor_quick.yaml

seed: 42
project: sensor_jepa
dataset:
name: cnc_milling
root: data/raw/sensor/cnc_milling
processed_dir: data/processed/sensor/cnc_milling
split: by_tool
window_size: 1024
stride: 512
normalize: per_channel_train
model:
encoder: conv1d
embedding_dim: 256
predictor_hidden_dim: 512
target_mode: shared
sigreg_weight: 0.05
masking:
temporal_mask_ratio: 0.4
channel_mask_ratio: 0.15
block_masking: true
pretrain:
epochs: 20
batch_size: 64
lr: 0.0003
weight_decay: 0.01
downstream:
task: classification
label_fraction: 0.1
probe_type: mlp
freeze_encoder: true
encoder_lr_scale: 0.0
eval:
metrics: ["accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"]
output_dir: outputs/sensor_jepa/cnc_milling_demo

Ejemplo Visual-JEPA quick:

configs/visual_jepa/demo_visual_quick.yaml

seed: 42
project: visual_jepa
dataset:
name: mvtec_ad
root: data/raw/visual/mvtec_ad
category: bottle
processed_dir: data/processed/visual/mvtec_ad
image_size: 224
train_only_normal: true
model:
encoder: small_vit
embedding_dim: 256
predictor_hidden_dim: 512
patch_size: 16
target_mode: shared
sigreg_weight: 0.05
masking:
mask_ratio: 0.6
block_masking: true
pretrain:
epochs: 20
batch_size: 32
lr: 0.0003
weight_decay: 0.01
anomaly:
scoring: latent_prediction_error
threshold_method: val_percentile
threshold_percentile: 95
eval:
metrics: ["image_auroc", "pixel_auroc", "auprc", "iou", "f1"]
output_dir: outputs/visual_jepa/mvtec_bottle_demo

────────────────────────────────────────
REPORTS COMERCIALES
────────────────────────────────────────

Crear dos reports:

outputs/reports/sensor_jepa_mvp_report.md
outputs/reports/visual_jepa_mvp_report.md

Sensor report debe incluir:

# Sensor-JEPA Predictive Maintenance MVP

## 1. Objetivo

Demostrar aprendizaje auto-supervisado sobre señales industriales no etiquetadas para mantenimiento predictivo.

## 2. Dataset

Nombre, sensores, número de ciclos/unidades, tarea downstream.

## 3. Método

JEPA temporal: masking, encoder, predictor, latent loss, SIGReg.

## 4. Modelos comparados

* baseline supervisado;
* frozen linear probe;
* frozen MLP probe;
* semi-frozen;
* full fine-tune.

## 5. Resultados

Tabla de métricas.

## 6. Visualizaciones

* embedding UMAP;
* wear/RUL curve;
* anomaly score timeline;
* confusion matrix.

## 7. Interpretación comercial

Qué significa para una fábrica:

* detectar desgaste;
* reducir inspección;
* priorizar mantenimiento;
* reducir etiquetado.

## 8. Limitaciones

* dataset público;
* no validado todavía con datos reales del cliente;
* requiere piloto con señales propias.

## 9. Propuesta de piloto

4-6 semanas, datos históricos, baseline, JEPA, dashboard, informe.

Visual report debe incluir:

# Visual-JEPA Industrial Quality MVP

## 1. Objetivo

Demostrar aprendizaje auto-supervisado para detectar defectos visuales industriales.

## 2. Dataset

MVTec/VisA/Kolektor, categoría, normales/anómalas, masks.

## 3. Método

JEPA visual: patch masking, encoder, predictor, latent loss.

## 4. Modelos comparados

* autoencoder/simple baseline;
* PatchCore/ResNet features si existe;
* Visual-JEPA latent anomaly;
* frozen probe;
* fine-tune.

## 5. Resultados

Image AUROC, pixel AUROC, AUPRC, IoU/F1.

## 6. Visualizaciones

* input;
* ground truth mask;
* heatmap;
* overlay;
* thresholded mask.

## 7. Interpretación comercial

Qué significa para control de calidad:

* entrenar con imágenes normales;
* detectar defectos raros;
* reducir ejemplos defectuosos necesarios;
* priorizar inspección humana.

## 8. Limitaciones

* dataset público;
* falta cámara real/línea real;
* calibración de falsos positivos necesaria.

## 9. Propuesta de piloto

Capturar imágenes normales, validar defectos, entrenar, threshold, dashboard.

────────────────────────────────────────
DASHBOARD / DEMO SIMPLE
────────────────────────────────────────

Opcional pero recomendado.

Crear:

scripts/18_launch_demo_app.py

Usar Streamlit si está disponible.

Dashboard Sensor-JEPA:

* elegir dataset;
* mostrar métricas;
* mostrar curva de desgaste/RUL;
* mostrar anomaly score;
* comparar baseline vs JEPA;
* mostrar embeddings.

Dashboard Visual-JEPA:

* elegir categoría;
* mostrar imagen;
* mostrar heatmap;
* mostrar overlay;
* mostrar score;
* mostrar threshold;
* mostrar top anomalías.

No bloquear MVP si Streamlit falla. El report Markdown debe ser suficiente.

────────────────────────────────────────
TESTS MÍNIMOS
────────────────────────────────────────

Implementar tests:

1. test_sensor_windowing.py

* entrada toy [T, C];
* window_size y stride;
* número correcto de ventanas;
* shape correcto.

2. test_visual_patching.py

* imagen toy;
* patch_size;
* número de patches;
* mask ratio aproximado.

3. test_freeze_encoder.py

* activar freeze;
* comprobar requires_grad=False en encoder;
* comprobar optimizer no recibe encoder params.

4. test_metrics.py

* classification metrics toy;
* anomaly AUROC toy;
* IoU mask toy.

5. test_dataset_manifest.py

* datasets.yaml existe;
* cada dataset tiene name, priority, local_path, task_types.

────────────────────────────────────────
COMANDOS DE USO
────────────────────────────────────────

README debe incluir comandos.

Preparar manifest:

python scripts/00_create_dataset_manifest.py

Preparar datos sensor:

python scripts/02_prepare_sensor_data.py --config configs/sensor_jepa/demo_sensor_quick.yaml

Pretrain sensor:

python scripts/04_pretrain_sensor_jepa.py --config configs/sensor_jepa/demo_sensor_quick.yaml

Probe sensor:

python scripts/05_probe_sensor_jepa.py --config configs/sensor_jepa/demo_sensor_quick.yaml --freeze-encoder --probe-type linear

Run sensor full demo:

python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml

Preparar datos visual:

python scripts/03_prepare_visual_data.py --config configs/visual_jepa/demo_visual_quick.yaml

Pretrain visual:

python scripts/08_pretrain_visual_jepa.py --config configs/visual_jepa/demo_visual_quick.yaml

Run visual full demo:

python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml

Run all:

python scripts/17_run_all_demos.py

────────────────────────────────────────
CRITERIOS DE ÉXITO DEL MVP
────────────────────────────────────────

Sensor-JEPA MVP se considera conseguido si:

* al menos un dataset sensor funciona end-to-end;
* hay pretraining JEPA;
* hay frozen probe;
* hay semi-frozen o full fine-tune;
* hay comparación contra baseline;
* hay métricas guardadas;
* hay report Markdown;
* hay al menos una visualización clara;
* se puede reproducir con un comando.

Visual-JEPA MVP se considera conseguido si:

* MVTec AD funciona end-to-end;
* hay pretraining JEPA con imágenes normales;
* hay anomaly heatmaps;
* hay métricas image-level y pixel-level;
* hay comparación contra baseline simple;
* hay report Markdown;
* hay overlays visuales;
* se puede reproducir con un comando.

No considerar terminado si:

* solo hay notebooks;
* no hay frozen probe;
* no hay comparación baseline vs JEPA;
* no hay outputs visuales;
* no hay README;
* no hay manifest de datasets;
* no hay reports reproducibles.

────────────────────────────────────────
NO HACER TODAVÍA
────────────────────────────────────────

No implementar todavía:

* arquitectura enorme tipo foundation model industrial;
* entrenamiento distribuido;
* cloud deployment;
* API SaaS;
* autenticación;
* dashboard complejo;
* integración con PLC real;
* integración con cámara real;
* modelos generativos;
* reconstrucción de imagen como objetivo principal;
* prometer biomasa, seguridad industrial o diagnóstico certificado.

Primero cerrar MVP defendible.

────────────────────────────────────────
ENTREGABLE FINAL QUE QUIERO DE TI
────────────────────────────────────────

Cuando termines, devuélveme:

1. Lista de archivos creados/modificados.
2. Datasets soportados y estado:

   * implementado;
   * parcial;
   * pendiente.
3. Comandos exactos para lanzar:

   * Sensor-JEPA demo;
   * Visual-JEPA demo.
4. Estructura de outputs generada.
5. Tabla de experimentos disponibles.
6. Métricas principales obtenidas si se ejecutaron.
7. Errores o limitaciones encontradas.
8. Qué dataset recomiendas usar primero para enseñar a una empresa.
9. Veredicto:

   * Sensor-JEPA MVP: No / Parcial / Sí.
   * Visual-JEPA MVP: No / Parcial / Sí.
10. Próximos pasos para convertirlo en piloto comercial.

Prioridad de implementación:

1. Manifest + README_DATASETS.
2. Loader CNC Milling Process Dataset.
3. Loader MVTec AD.
4. Sensor-JEPA mínimo.
5. Visual-JEPA mínimo.
6. Frozen probe para ambos.
7. Baselines simples.
8. Comparación automática.
9. Reports Markdown.
10. Visualizaciones.
11. Tests.
12. Streamlit demo opcional.

Haz cambios pequeños, limpios y reproducibles. No inventes resultados. Si no puedes descargar un dataset por login/licencia, deja instrucciones claras y valida rutas locales.

────────────────────────────────────────
BENCHMARK & VALIDATION LAYER
────────────────────────────────────────

Añadir una capa rigurosa de validación para saber si Sensor-JEPA y Visual-JEPA realmente aportan valor frente a modelos clásicos, modelos profundos normales y baselines fuertes del área.

Objetivo:

No quiero solo entrenar JEPA. Quiero saber si JEPA mejora o no mejora respecto a:

1. Modelos simples:

   * MLP;
   * Logistic Regression / Ridge;
   * Random Forest;
   * XGBoost o LightGBM si están disponibles.

2. Modelos convolucionales o deep learning clásicos:

   * 1D CNN para sensores;
   * TCN si es sencillo;
   * LSTM/GRU para sensores;
   * CNN/ResNet supervisada para imágenes;
   * Autoencoder visual;
   * Autoencoder temporal.

3. Baselines fuertes del dominio:

   * ROCKET o MiniROCKET para clasificación de series temporales;
   * TS2Vec o equivalente auto-supervisado para series temporales si se puede integrar;
   * PatchCore para anomalía visual industrial;
   * PaDiM para anomalía visual industrial;
   * ResNet features + kNN como baseline visual simple fuerte.

4. Variantes JEPA:

   * JEPA frozen linear probe;
   * JEPA frozen MLP probe;
   * JEPA semi-frozen;
   * JEPA full fine-tune;
   * JEPA con 1%, 5%, 10%, 25% y 100% de etiquetas.

Objetivo experimental:

Responder de forma honesta a estas preguntas:

A. ¿JEPA mejora frente a MLP/CNN clásico?
B. ¿JEPA mejora cuando hay pocas etiquetas?
C. ¿JEPA aprende representaciones reutilizables con encoder congelado?
D. ¿JEPA compite contra baselines fuertes del dominio?
E. ¿JEPA solo funciona cuando se hace full fine-tuning?
F. ¿JEPA es más estable entre semillas?
G. ¿JEPA reduce el coste de etiquetado?
H. ¿JEPA mejora anomalías raras o clases minoritarias?

────────────────────────────────────────
VALIDACIÓN PARA SENSOR-JEPA
────────────────────────────────────────

Crear un benchmark específico para sensores:

scripts/18_benchmark_sensor_models.py

Debe comparar, como mínimo:

1. MLP sobre features procesadas.
2. Random Forest sobre features procesadas.
3. XGBoost/LightGBM si está disponible; si no, fallback a HistGradientBoostingClassifier/Regressor de sklearn.
4. 1D CNN supervisada desde cero.
5. LSTM o GRU supervisado.
6. Autoencoder temporal + anomaly score.
7. Isolation Forest sobre features.
8. ROCKET/MiniROCKET si se puede instalar o implementar mediante sktime.
9. TS2Vec o baseline auto-supervisado equivalente si es razonable integrarlo.
10. Sensor-JEPA frozen linear.
11. Sensor-JEPA frozen MLP.
12. Sensor-JEPA semi-frozen.
13. Sensor-JEPA full fine-tune.

Tareas mínimas:

Para CNC / tool wear:

* Wear_Class classification;
* VB_mm regression si existe;
* anomaly detection usando Healthy/early-life como normal;
* RUL proxy si hay ciclo de vida completo.

Para bearing datasets:

* normal vs fault;
* fault type classification;
* transfer across operating conditions;
* anomaly detection.

Para C-MAPSS:

* RUL regression;
* health stage classification.

Métricas:

Classification:

* accuracy;
* balanced accuracy;
* macro-F1;
* weighted-F1;
* per-class F1;
* confusion matrix.

Regression:

* MAE;
* RMSE;
* R2;
* Spearman correlation.

RUL:

* MAE;
* RMSE;
* NASA score si está implementado;
* error early-life;
* error late-life.

Anomaly:

* AUROC;
* AUPRC;
* precision@k;
* false alarm rate;
* detection lead time si se puede calcular.

Few-label evaluation:

Ejecutar todos los modelos compatibles con estas fracciones de etiquetas:

* 1%;
* 5%;
* 10%;
* 25%;
* 100%.

Para cada label_fraction, mantener el mismo split y la misma seed.

Seeds:

Ejecutar mínimo 3 semillas:

* 42;
* 123;
* 999.

Si el coste es alto, permitir flag:

--quick

que ejecute solo 1 seed y menos epochs.

Outputs:

outputs/sensor_jepa/benchmark/
sensor_benchmark_results.csv
sensor_benchmark_results.md
sensor_benchmark_summary.json
plots/
metric_vs_label_fraction.png
model_ranking_macro_f1.png
model_ranking_auroc.png
rul_prediction_curves.png
anomaly_score_curves.png
confusion_matrices/
reports/
sensor_benchmark_report.md

Columnas mínimas de sensor_benchmark_results.csv:

* dataset;
* task;
* model_name;
* model_family;
* seed;
* label_fraction;
* encoder_mode;
* probe_type;
* frozen_encoder;
* trainable_params;
* accuracy;
* balanced_accuracy;
* macro_F1;
* weighted_F1;
* MAE;
* RMSE;
* R2;
* AUROC;
* AUPRC;
* precision_at_k;
* train_time_sec;
* inference_time_ms;
* notes.

Interpretación automática:

El report debe incluir:

1. Mejor modelo global por métrica.
2. Mejor modelo con pocas etiquetas.
3. Mejor modelo no profundo.
4. Mejor modelo profundo clásico.
5. Mejor variante JEPA.
6. Si JEPA supera a MLP.
7. Si JEPA supera a CNN/LSTM.
8. Si JEPA supera a ROCKET/MiniROCKET.
9. Si JEPA mejora especialmente con pocas etiquetas.
10. Si JEPA no mejora, explicar posibles razones.

Reglas de conclusión:

* Si JEPA frozen supera a MLP/CNN y es competitivo con ROCKET/MiniROCKET:
  “Sensor-JEPA aprende representaciones reutilizables y es competitivo con baselines fuertes.”

* Si JEPA solo gana con pocas etiquetas:
  “Sensor-JEPA aporta valor principalmente en régimen low-label.”

* Si JEPA solo gana en full fine-tune:
  “Sensor-JEPA ayuda como inicialización, pero no está demostrado que el embedding congelado sea superior.”

* Si JEPA pierde contra baselines simples:
  “No hay evidencia suficiente de valor; revisar arquitectura, masking, normalización o splits.”

────────────────────────────────────────
VALIDACIÓN PARA VISUAL-JEPA
────────────────────────────────────────

Crear un benchmark específico para visión industrial:

scripts/19_benchmark_visual_models.py

Debe comparar, como mínimo:

1. Autoencoder convolucional.
2. ResNet features + kNN.
3. PaDiM si se puede integrar.
4. PatchCore si se puede integrar.
5. CNN/ResNet supervisada con pocas etiquetas.
6. UNet segmentation si hay masks y es sencillo.
7. Visual-JEPA latent prediction error.
8. Visual-JEPA frozen linear probe.
9. Visual-JEPA frozen MLP probe.
10. Visual-JEPA semi-frozen.
11. Visual-JEPA full fine-tune.

Datasets mínimos:

* MVTec AD, mínimo categorías:

  * bottle;
  * metal_nut;
  * tile;
  * wood;
  * transistor.

* Si hay tiempo:

  * VisA;
  * KolektorSDD;
  * NEU Surface Defect;
  * Severstal.

Métricas:

Image-level anomaly:

* AUROC;
* AUPRC;
* F1;
* accuracy;
* false positive rate;
* threshold usado.

Pixel-level anomaly:

* pixel AUROC;
* pixel AUPRC;
* IoU;
* Dice/F1;
* PRO score opcional.

Few-shot classification:

* balanced accuracy;
* macro-F1;
* per-class F1.

Localization:

* mean IoU;
* Dice;
* visual quality of heatmaps.

Few-label evaluation:

Para modelos supervisados o probes, evaluar:

* 1%;
* 5%;
* 10%;
* 25%;
* 100%.

Para anomaly detection one-class:

* entrenar solo con normales;
* evaluar normales + defectos.

Seeds:

Ejecutar mínimo 3 semillas:

* 42;
* 123;
* 999.

Outputs:

outputs/visual_jepa/benchmark/
visual_benchmark_results.csv
visual_benchmark_results.md
visual_benchmark_summary.json
plots/
image_auroc_by_model.png
pixel_auroc_by_model.png
metric_vs_label_fraction.png
model_ranking_by_category.png
heatmaps/ <category>/
sample_001_input.png
sample_001_gt_mask.png
sample_001_patchcore_overlay.png
sample_001_padim_overlay.png
sample_001_jepa_overlay.png
reports/
visual_benchmark_report.md

Columnas mínimas de visual_benchmark_results.csv:

* dataset;
* category;
* model_name;
* model_family;
* seed;
* train_mode;
* label_fraction;
* anomaly_method;
* image_AUROC;
* image_AUPRC;
* pixel_AUROC;
* pixel_AUPRC;
* pixel_IoU;
* pixel_F1;
* threshold;
* train_time_sec;
* inference_time_ms;
* trainable_params;
* notes.

Interpretación automática:

El report debe incluir:

1. Mejor modelo image-level.
2. Mejor modelo pixel-level.
3. Mejor modelo por categoría.
4. Mejor modelo con pocas etiquetas.
5. Mejor baseline clásico.
6. Mejor baseline fuerte.
7. Mejor variante JEPA.
8. Si JEPA supera a autoencoder.
9. Si JEPA supera a ResNet+kNN.
10. Si JEPA se acerca o supera a PatchCore/PaDiM.
11. Casos donde JEPA falla.
12. Casos donde JEPA produce mejores heatmaps visuales.

Reglas de conclusión:

* Si JEPA supera autoencoder y ResNet+kNN:
  “Visual-JEPA aporta valor frente a baselines simples.”

* Si JEPA se acerca a PatchCore/PaDiM:
  “Visual-JEPA es competitivo con baselines fuertes.”

* Si JEPA supera PatchCore/PaDiM:
  “Resultado potencialmente publicable; repetir con más seeds, más categorías y comparación con números publicados.”

* Si JEPA solo mejora en few-shot:
  “Visual-JEPA aporta valor como pretraining low-label.”

* Si JEPA pierde claramente:
  “No afirmar mejora; usar JEPA como exploración o revisar patching/masking/scoring.”

────────────────────────────────────────
NO HACER CLAIM SOTA SIN VALIDACIÓN
────────────────────────────────────────

Añadir al README y a los reports una advertencia:

No afirmar “SOTA” salvo que se cumplan todas estas condiciones:

1. Comparación contra baselines fuertes del área.
2. Mismos datasets y splits estándar.
3. Mismas métricas que la literatura.
4. Mínimo 3 semillas o estabilidad equivalente.
5. Test set no usado para tuning.
6. Threshold elegido solo en validación.
7. Resultados reproducibles con comandos.
8. Comparación con números publicados.
9. Report de tiempo de entrenamiento/inferencia.
10. Código y configs guardados.

Si no se cumplen, usar lenguaje:

* “MVP competitivo frente a baselines internos.”
* “Prometedor en régimen low-label.”
* “Mejora frente a MLP/CNN baseline.”
* “Competitivo con baseline clásico.”
* “No validado como SOTA.”

────────────────────────────────────────
CRITERIOS DE ÉXITO REALES
────────────────────────────────────────

Nivel 1 — MVP comercial:

* JEPA supera MLP/Autoencoder/CNN simple o mejora con pocas etiquetas.
* Hay report y visualizaciones.
* Se puede enseñar a empresas.

Nivel 2 — MVP técnico fuerte:

* JEPA compite con ROCKET/MiniROCKET en sensores.
* JEPA compite con PaDiM/PatchCore en visión.
* Hay varias semillas.
* Hay frozen probe positivo.

Nivel 3 — Posible paper/SOTA:

* JEPA supera baselines fuertes bajo protocolo estándar.
* Hay comparación con literatura.
* Hay ablations.
* Hay robustez multi-dataset.
* Hay análisis estadístico.

No priorizar Nivel 3 hasta terminar Nivel 1 y Nivel 2.

────────────────────────────────────────
ANÁLISIS ESTADÍSTICO
────────────────────────────────────────

Implementar análisis simple:

* media ± desviación estándar por modelo;
* ranking por métrica;
* diferencia absoluta frente a baseline;
* diferencia relativa frente a baseline;
* intervalo de confianza bootstrap opcional;
* test pareado opcional si hay varias seeds o folds.

En los CSV añadir:

* mean_metric;
* std_metric;
* delta_vs_baseline;
* relative_delta_vs_baseline;
* rank.

Generar una tabla:

outputs/<project>/benchmark/model_ranking.md

Con columnas:

* rank;
* model;
* metric_mean;
* metric_std;
* delta_vs_baseline;
* conclusion.

────────────────────────────────────────
ABLATIONS MÍNIMAS JEPA
────────────────────────────────────────

Añadir ablations para saber por qué JEPA funciona o no:

Sensor-JEPA:

* temporal_mask_ratio: 0.2, 0.4, 0.6;
* channel_mask_ratio: 0.0, 0.15, 0.3;
* embedding_dim: 128, 256;
* encoder: conv1d vs transformer;
* sigreg_weight: 0.0, 0.05.

Visual-JEPA:

* mask_ratio: 0.4, 0.6, 0.75;
* patch_size: 8, 16, 32;
* encoder: small_conv vs small_vit;
* sigreg_weight: 0.0, 0.05;
* scoring: latent_prediction_error vs kNN memory.

No ejecutar todas las combinaciones por defecto. Crear modo:

--ablation quick

que ejecute solo las más importantes.

────────────────────────────────────────
ENTREGABLE FINAL DEL BENCHMARK
────────────────────────────────────────

Al terminar, devolver:

1. Qué baselines se implementaron realmente.
2. Qué baselines quedaron pendientes y por qué.
3. Si JEPA supera MLP/CNN clásico.
4. Si JEPA supera o se acerca a baselines fuertes.
5. En qué dataset/tarea funciona mejor.
6. En qué dataset/tarea falla.
7. Si hay evidencia de mejora low-label.
8. Si hay evidencia de representación reutilizable con frozen probe.
9. Si se puede vender como MVP.
10. Si se puede o no hacer claim de SOTA.


Actualiza el prompt/manifest según las rutas reales ya descargadas:

Datasets disponibles y prioritarios para Fase 1:

SENSOR:
1. cnc_milling
   local_path: data/raw/sensor/cnc_milling
   files:
   - FeatureAndMetadata_Milling.csv
   - metadata.xlsx
   - raw_data/*.csv
   raw naming pattern real:
   - raw_data/P*_F*_C*.csv
   count esperado:
   - 968 CSV raw

2. cwru_bearing
   local_path: data/raw/sensor/cwru_bearing
   estructura real:
   - normal/
   - 12k_drive_end/
   - 48k_drive_end/
   count esperado:
   - 111 archivos .mat

3. paderborn_bearing
   local_path: data/raw/sensor/paderborn_bearing
   estructura real:
   - K001/
   - K002/
   - ...
   - KA*/
   - KB*/
   - KI*/
   count esperado:
   - 2560 archivos .mat

VISUAL:
4. mvtec_ad
   local_path: data/raw/visual/mvtec_ad
   estructura real:
   - bottle/
   - cable/
   - capsule/
   - carpet/
   - grid/
   - hazelnut/
   - leather/
   - metal_nut/
   - pill/
   - screw/
   - tile/
   - toothbrush/
   - transistor/
   - wood/
   - zipper/
   count esperado:
   - 15 categorías

5. visa
   local_path: data/raw/visual/visa
   estructura real:
   - candle/
   - capsules/
   - cashew/
   - chewinggum/
   - fryum/
   - macaroni1/
   - macaroni2/
   - pcb1/
   - pcb2/
   - pcb3/
   - pcb4/
   - pipe_fryum/
   - split_csv/
   - LICENSE-DATASET

6. kolektor_sdd
   local_path: data/raw/visual/kolektor_sdd
   estructura real:
   - KolektorSDD-boxes/kos01/
   - ...
   - KolektorSDD-boxes/kos50/
   dentro de cada kosXX:
   - PartN.jpg
   - PartN_label.bmp
   count esperado:
   - 50 carpetas kos

Datasets pendientes / no implementar en Fase 1 salvo que existan archivos reales:
- multi_sensor_cnc
- nasa_ims_bearing
- cmapss
- ai4i
- mvtec_3d_ad
- neu_surface
- dagm_2007
- severstal
- wood_surface_defects

Regla:
No fallar si esos datasets pendientes no tienen archivos. Marcar estado como "missing" o "pending" en data/manifests/datasets.yaml y continuar con los datasets disponibles.