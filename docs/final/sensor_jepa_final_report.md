# Sensor-JEPA and World Model: Final Technical Report

## 1. Resumen Ejecutivo (Executive Summary)
Este informe consolida los resultados y hallazgos del desarrollo del MVP de Sensor-JEPA y el modelo del mundo latente condicionado por acciones para la predicción de fallos inminentes en maquinaria industrial (principalmente fresadoras CNC). El desarrollo partió de una hipótesis ambiciosa de alcanzar rendimiento SOTA empleando representaciones latentes predictivas autosupervisadas. Sin embargo, la posterior **validación adversarial rigurosa** reveló que los indicadores operativos (proxies de ciclo y metadatos) dominaban el rendimiento del modelo en los esquemas de validación estándar.

Al aislar el efecto de estos proxies, demostramos que las señales sensoriales y las representaciones latentes predictivas aportan un valor incremental robusto en escenarios de generalización difícil ("no-cycle" y "held-out hardness"). No obstante, no existe evidencia que respalde que JEPA supere de forma sistemática y robusta a las características sensoriales tradicionales de ingeniería (engineered features). Recomendamos un producto híbrido basado en metadatos y características de ingeniería física, dejando los componentes JEPA/World-Model como módulos complementarios.

## 2. Datasets de Sensor
* **CNC Milling (Completo)**: Dataset principal de fresado de la NASA. Contiene variables físicas de corrientes de motores, vibraciones y emisión acústica muestreadas a alta frecuencia ($250\,\text{Hz}$), estructuradas en ciclos de fresado sobre diversos materiales (acero, cera) con diferentes herramientas (cortadores). Cuenta con etiquetas de desgaste del cortador y ciclos restantes hasta el fallo.
* **CWRU Bearing (Parcial/Estructura de Datos)**: Dataset de vibraciones de rodamientos de la Case Western Reserve University. Implementado en la estructura de carga e inspección de manifiestos, listo para expandirse a pruebas de transferencia.
* **Paderborn Bearing (Parcial/Estructura de Datos)**: Dataset de vibraciones de rodamientos de la Universidad de Paderborn. Implementado a nivel de manifiesto para futura experimentación en diagnóstico y pronóstico.

## 3. Arquitecturas Implementadas
1. **Sensor-JEPA Global**: Codificador Conv1D que procesa ventanas temporales de señales multicanal proyectándolas en un espacio latente $z_t$. La optimización se realiza mediante una pérdida de predicción latente y regularización de covarianza y varianza (VICReg-style) para evitar el colapso del representación.
2. **Sensor World Model (LeWorldModel)**: Red predictiva que toma la representación latente $z_t$ y un vector de contexto o acción $a_t$ (parámetros de corte del CNC) para pronosticar el estado latente futuro $\hat{z}_{t+h}$. Un clasificador de riesgo (sonda logística) lee la diferencia o la propia representación futura $\hat{z}_{t+h}$ para emitir un score de riesgo de fallo próximo (`CycleToFailure <= K`).
3. **Incremental Value Benchmark**: Capa evaluativa que entrena estimadores lineales o de árbol comparando la mejora incremental de rendimiento al añadir características en grupos emparejados (ej. `metadata + sensor` vs `metadata_only`).
4. **DenseSensorJEPA (Experimental)**: Arquitectura que divide la serie temporal en "tokens" temporales (parches) y entrena un predictor de parches enmascarados con codificador de destino EMA, calculando curvas de sorpresa temporal y anomalías locales.

## 4. Validación Adversarial
Para evitar claiming espurio, el MVP incluye auditorías estrictas:
* **Fuga de datos (Leakage check)**: Garantiza que las herramientas (herramientas físicas con IDs concretos) del conjunto de test nunca hayan sido vistas en el entrenamiento.
* **Auditoría de características (Feature audit)**: Asegura que ni el codificador JEPA ni los modelos predictivos usen directa o indirectamente variables prohibidas como RUL, números de ciclo absoluto o variables del ciclo futuro.
* **Líneas base específicas**:
  * *Metadata-only*: Clasificadores que solo usan datos de configuración (tipo de herramienta, holder, material, etc.).
  * *Cycle-only*: Modelos basados únicamente en el índice del ciclo actual, que actúa como proxy de desgaste acumulado.
  * *Sensor-only/JEPA-only*: Evaluados sin acceso a metadatos de ciclo para medir el valor físico puro de la señal.

## 5. Resultados Principales (Tablas de Evidencia)

A continuación se presentan los resultados consolidados de las fases experimentales más rigurosas de validación.

### 5.1. Protocolo Sin Proxies de Ciclo (No-Cycle Agregado)
Mide el rendimiento (AUPRC promedio sobre 3 semillas) cuando se prohíbe el uso de índices de ciclo que actúan como proxies triviales de desgaste temporal.

| Modelo / Características | AUPRC Medio | Delta vs Metadata |
| :--- | :---: | :---: |
| `metadata_only_no_cycle` | 0.2386 | +0.0000 |
| `current_z_only` | 0.4451 | +0.2065 |
| `sensor_raw_only` | 0.5200 | +0.2814 |
| `metadata_plus_current_z_plus_predicted_future_z` | 0.5370 | +0.2983 |
| `predicted_future_z_only` | 0.5502 | +0.3115 |
| `sensor_engineered_only` | 0.5583 | +0.3197 |

### 5.2. Generalización Difícil (Hard Generalization)
Mide la robustez en splits desafiantes donde ciertas clases de dureza de material, condiciones de corte o herramientas específicas quedan completamente fuera del entrenamiento (held-out).

| Split / Protocolo | Modelo / Configuración | AUPRC Medio |
| :--- | :--- | :---: |
| **held_out_hardness_bin** | `metadata_only` | 0.5128 |
| | `current_z_only` | 0.6638 |
| | `metadata_plus_current_z_plus_predicted_future_z` | 0.7150 |
| | `sensor_engineered_only` | 0.7631 |
| **held_out_cutting_condition** | `sensor_raw_only` | 0.6290 |
| | `current_z_only` | 0.7146 |
| | `metadata_only` | 0.7182 |
| | `metadata_plus_predicted_future_z` | 0.7188 |
| **held_out_tool_id** | `current_z_only` | 0.3604 |
| | `sensor_raw_only` | 0.5742 |
| | `metadata_only` | 0.6273 |

### 5.3. Comparativa: JEPA vs Características de Ingeniería (Engineered)
Analiza si las representaciones latentes autosupervisadas de JEPA o del modelo del mundo aportan valor más allá de las características físicas calculadas manualmente (como RMS, energía por bandas, curtosis, etc.).

*Métricas evaluadas en el protocolo no-cycle:*

| Configuración de Características | AUPRC Medio | Delta vs Sensor Engineered |
| :--- | :---: | :---: |
| `predicted_future_z_only` | 0.5502 ± 0.2022 | -0.0082 |
| `sensor_engineered_plus_predicted_future_z` | 0.5670 ± 0.2513 | +0.0087 |
| `sensor_engineered_plus_current_z` | 0.5733 ± 0.2690 | +0.0150 |

### 5.4. Comparativa en Dureza Excluida (Held-out Hardness: JEPA vs Engineered)
Resultados específicos bajo el split de exclusión de dureza (`held_out_hardness_bin`):

| Modelo | AUPRC Medio |
| :--- | :---: |
| `predicted_future_z_only` | 0.5540 ± 0.2027 |
| `current_z_only` | 0.5816 ± 0.1863 |
| `sensor_engineered_only` | 0.6923 ± 0.1685 |
| `sensor_engineered_plus_current_z` | 0.7052 ± 0.1723 |
| `sensor_engineered_plus_current_z_plus_predicted_future_z` | 0.7201 ± 0.1741 |
| `metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z` | 0.7269 ± 0.1694 |

### 5.5. DenseSensor: Diagnóstico de Sorpresa Local
Evaluación experimental de DenseSensorJEPA empleando agrupamiento de embeddings (pooling) frente a metadatos puros:

| Modelo | AUPRC | AUROC |
| :--- | :---: | :---: |
| `dense_sensor_negative_topk_surprise` | 0.1700 | - |
| `dense_embedding_pool_mlp` | 0.2002 | 0.7481 |
| `metadata_plus_dense_embedding_pool_mlp` | 0.3442 | 0.7506 |
| `metadata_only` | 0.6609 | - |
| `metadata_plus_surprise` | 0.6671 | - |

## 6. Interpretación Técnica
1. **Dominancia de Metadatos y Ciclo en Operational**: En el protocolo estándar de fábrica, la posición del ciclo actual domina completamente el pronóstico. Es un proxy directo del desgaste acumulativo. Por lo tanto, no se requiere inteligencia compleja para puntuar el riesgo operacional básico.
2. **Valor Sensorial en Escenarios Complejos**: Cuando eliminamos proxies de ciclo (`no_cycle`), el AUPRC de los metadatos puros cae a $0.2386$, mientras que las señales sensoriales de ingeniería suben el rendimiento a $0.5583$ (+0.3197) y las latentes futuras predictivas de JEPA alcanzan $0.5502$ (+0.3115). Esto demuestra que los sensores importan cuando el ciclo no está disponible o no es un proxy lineal fiable.
3. **Generalización por Dureza (Hardness Split)**: El split `held_out_hardness_bin` es el más prometedor para los sensores. El uso de características combinadas de ingeniería y latentes JEPA eleva el AUPRC de $0.6923$ a $0.7269$ (+0.0346), lo que sugiere complementariedad en regímenes de corte difíciles.
4. **Insuficiencia del Diagnóstico de DenseSensor**: DenseSensorJEPA por sí mismo no ofrece valor predictor útil actualmente ($0.2002$ AUPRC vs $0.6609$ del baseline). Requiere rediseño del agrupador y la máscara de entrenamiento.

## 7. Claims Permitidos y Prohibidos
* **PERMITIDO**: "El MVP proporciona un andamiaje técnico de análisis de riesgo (risk scoring) industrial libre de fugas de datos."
* **PERMITIDO**: "Las características físicas basadas en señales de sensores (tanto clásicas como latentes autosupervisadas) aportan un valor incremental significativo de hasta +0.31 de AUPRC cuando no se dispone de proxies de ciclo lineal."
* **PROHIBIDO**: "Sensor-JEPA es un modelo SOTA que supera a todas las líneas base." (No se cuenta con soporte oficial de MiniROCKET o TS2Vec para afirmarlo).
* **PROHIBIDO**: "El modelo del mundo aprende relaciones causales de las acciones." (El modelo solo captura correlaciones condicionales dentro del dataset CNC).

## 8. Implicaciones Comerciales
Para la venta de este MVP a clientes, se debe evitar el "hype" de JEPA. Comercialmente:
1. El modelo base debe apoyarse en **metadatos operativos y ciclo acumulado**, que son muy baratos de capturar y altamente eficaces.
2. El procesamiento de **sensores físicos** se venderá como un módulo de seguridad adicional ("Safe Generalization") para detectar anomalías cuando la máquina trabaja con materiales inusuales o herramientas nuevas.
3. Las representaciones **JEPA y World Model** se activarán únicamente como un módulo experimental de optimización avanzada.

## 9. Siguientes Pasos
1. Integrar las librerías oficiales de series temporales (`aeon`, `sktime`) para comparativas definitivas de baselines académicos.
2. Rediseñar el scoring de sorpresa local en DenseSensorJEPA, utilizando pérdida basada en distancias ponderadas.
3. Expandir la validación de transferencia de representaciones latentes a los datasets de rodamientos Paderborn y CWRU.
