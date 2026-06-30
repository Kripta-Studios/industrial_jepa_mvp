# Posicionamiento Comercial del MVP de Inteligencia Artificial Industrial

Este documento interno define la estrategia de comercialización del MVP del proyecto `industrial_jepa_mvp`. Establece qué podemos vender de forma honesta, qué queda excluido, cómo estructurar un proyecto piloto y cómo alinear la oferta con las principales industrias y empresas de Galicia.

---

## 1. ¿Qué se puede vender honestamente hoy?
Nuestra propuesta de valor debe alejarse de la propaganda tecnológica e invenciones académicas. Con el MVP actual, podemos vender de forma honesta:
1. **Andamiaje de Auditoría de Fugas (Anti-Leakage Audit)**: Un framework de análisis riguroso que detecta si los modelos actuales del cliente están sobreestimando su precisión debido a proxies operacionales (ej. ciclos acumulados) o variables con fuga de información.
2. **Modelado de Riesgo Basado en Metadatos y Sensores Físicos (CNC/Procesos)**: Un predictor calibrado de riesgo de fallo inminente (`CycleToFailure <= K`) que combina metadatos del proceso con variables de ingeniería sensorial clásica.
3. **Módulo Predictivo Latente (Experimental)**: El modelo del mundo latente (Sensor World Model) que proyecta las señales a un espacio abstracto y calcula la sorpresa ante cambios de comportamiento. Se vende como un módulo de optimización opcional, activado solo si incrementa la precisión en la validación local del cliente.

## 2. ¿Qué NO se puede vender (Exclusiones)?
* **Rendimiento SOTA Generalizado**: No podemos prometer superar a cualquier baseline comercial o académico sin un benchmark específico con los datos del cliente.
* **Modelos del Mundo Causales**: El modelo predice secuencias latentes basadas en correlaciones del proceso, pero no descubre leyes de causalidad física pura.
* **Control de Calidad Visual Listo para Producción**: El MVP de visión actual (Visual-JEPA global) es débil y no compite con métodos comerciales. Se presenta como un roadmap de desarrollo en fase de planificación técnica, no como un producto desplegable.

---

## 3. Estructura del Piloto Comercial (4-6 semanas)
Proponemos a las empresas gallegas un piloto acotado para evaluar la viabilidad del mantenimiento predictivo en sus plantas sin grandes inversiones iniciales:

* **Semana 1-2: Ingesta de Datos y Diagnóstico de Fugas**:
  * Recopilación de metadatos (órdenes de trabajo, ciclos, herramientas) e históricos de sensores.
  * Ejecución de la auditoría adversarial para comprobar la presencia de proxies triviales.
* **Semana 3-4: Entrenamiento de Modelos e Ingeniería de Características**:
  * Configuración de la línea base con características físicas de ingeniería (RMS, análisis de espectro) y metadatos.
  * Entrenamiento del modelo predictivo del mundo latente sobre los datos locales.
* **Semana 5: Validación Cruzada Generalizada (Hard Generalization)**:
  * Evaluación en herramientas o condiciones operativas excluidas para simular fallos imprevistos reales.
  * Cálculo del valor incremental real aportado por el modelo.
* **Semana 6: Entrega de Resultados y Dashboard Comercial**:
  * Entrega de un reporte técnico final detallando el AUPRC y la reducción de falsas alarmas.
  * Demostración interactiva en HTML/CSS basada en los datos del cliente y hoja de ruta de despliegue.

---

## 4. Requerimientos de Ingesta (Inputs y Outputs)

### Inputs necesarios de la empresa cliente:
* **Histórico de Variables de Proceso**: Muestreos de sensores de corriente, vibración, temperatura o SCADA (frecuencia recomendada $> 100\,\text{Hz}$).
* **Metadatos Operativos**: Registro de configuraciones de máquina, tipos de material procesado, dimensiones de herramientas o lotes de producción.
* **Registro de Fallos / Mantenimiento**: Fechas y horas exactas de rotura, paradas por desgaste, cambios preventivos o sustitución de consumibles.

### Outputs entregados al cliente:
* **Reporte de Auditoría de Datos**: Informe detallando la calidad de sus históricos y alertas de fuga de proxies.
* **Modelo Calibrado de Risk Scoring**: Algoritmo ajustado para emitir probabilidades reales de fallo.
* **Dashboard Demo Standalone**: Interfaz visual (HTML interactivo) para la simulación de alertas de mantenimiento y priorización de inspecciones.

---

## 5. Riesgos del Proyecto
* **Falta de Historial de Fallos**: Muchas industrias tienen muy pocos fallos reales registrados por políticas estrictas de mantenimiento preventivo, lo que dificulta entrenar clasificadores de riesgo. Se mitiga orientando el modelo hacia la detección de anomalías sin supervisión.
* **Sensores Inestables o Descalibrados**: Cambios de sensorización en planta pueden alterar el espacio latente. Requiere calibración periódica.

---

## 6. Sectores Objetivo en Galicia y Prioridades de Pilotaje

El tejido industrial gallego ofrece oportunidades clave donde encajan nuestras dos líneas de desarrollo. Proponemos las siguientes empresas de ejemplo sin asumir relaciones ni contactos previos:

1. **FINSA (Madera / Tableros)**: *Prioridad 1*.
   * *Encaje*: Control de calidad visual de tableros (nudos, grietas) y monitorización de rodillos, prensas y sierras con sensores SCADA.
2. **Televés (Electrónica / Telecomunicaciones)**: *Prioridad 2*.
   * *Encaje*: Inspección de soldadura en PCBs y logs de pruebas de radiofrecuencia para predecir fallos en líneas SMT.
3. **Marine Instruments (Sistemas Marinos / Boyas Satelitarias)**: *Prioridad 3*.
   * *Encaje*: Telemetría satelital y algoritmos de anomalías en boyas y sistemas marinos bajo entornos extremos.
4. **Norvento (Energía / Eólica)**: *Prioridad 4*.
   * *Encaje*: Análisis de vibraciones en multiplicadoras de aerogeneradores y datos SCADA de parques eólicos para optimización del mantenimiento.
5. **Cinfo (Vídeo Inteligente / IA)**: *Prioridad 5*.
   * *Encaje*: Procesamiento inteligente de flujos de vídeo para seguridad o monitorización de eventos visuales anomalías.
6. **Alén Space (Espacial / Telemetría)**: *Prioridad 6*.
   * *Encaje*: Análisis de series temporales de satélites (cubesats) para detección de anomalías térmicas o eléctricas en órbita.
7. **Jealsa / Frinsa (Alimentación / Conservas)**: *Prioridad 7*.
   * *Encaje*: Inspección de latas, sellado de envases y monitorización de autoclaves y cintas de transporte.
8. **Hijos de Rivera (Alimentación / Bebidas)**: *Prioridad 8*.
   * *Encaje*: Control de procesos de embotellado y mantenimiento preventivo de bombas y motores de fermentación.
9. **Zendal / CZ Vaccines (Bioprocesos / Farmacéutico)**:
   * *Encaje*: Desviaciones tempranas en series temporales de biorreactores y mantenimiento de autoclaves estériles.
10. **Nueva Pescanova (Acuicultura / Procesamiento)**:
    * *Encaje*: Monitorización de turbidez y oxígeno en acuicultura y control visual de fileteado y envasado de pescado.
11. **Navantia (Astilleros / Construcción Naval)**:
    * *Encaje*: Trazabilidad en soldaduras de bloques y mantenimiento predictivo de grúas pórtico y maquinaria pesada de astillero.
12. **ABANCA (Financiero / Datos)**:
    * *Encaje* (Extensión de negocio no físico): Detección de patrones anómalos en flujos transaccionales y ciberseguridad.
13. **Inditex (Logística / Textil)**: *Objetivo a largo plazo*.
    * *Encaje*: Inspección visual automática de prendas, gestión de stocks con visión y optimización de clasificadores logísticos.
