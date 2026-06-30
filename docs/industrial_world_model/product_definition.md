# Product Definition

## Product Name

Recommended positioning:

`Industrial Predictive Quality World Model`

Alternative:

`Hierarchical Industrial World Model for Predictive Quality and Anomaly Detection`

## What Is Sold

A 4-6 week pilot that validates whether a client's industrial data can support:

- visual anomaly detection with DINOv2/PatchCore/PaDiM foundation;
- predictive quality risk scoring;
- sensor/process anomaly scoring;
- early warning from process state and setpoints;
- patch/image/cycle/lot/line risk aggregation.

## Inputs Required

- images or video frames, if visual quality is in scope;
- sensor histories or process time series;
- process context: recipe, material, speed, feed, temperature, pressure, setpoints;
- labels, defects, rework, scrap, failures or inspection outcomes;
- grouping metadata: piece, cycle, batch, line, shift or machine.

## Outputs Delivered

- dataset manifest;
- baseline benchmark;
- visual anomaly heatmaps;
- risk ranking;
- top-10 alerts;
- report with what works and what does not;
- local HTML demo;
- go/no-go recommendation for next phase.

## Difference From Previous Sensor MVP

The previous sensor line focused on CNC failure-soon forecasting. This new line expands the product scope to predictive quality:

- visual + sensor + process inputs;
- dense visual features;
- LeJEPA/SIGReg in-domain learning;
- latent world model;
- hierarchical alerting.

## What Is Not Sold

- autonomous plant control;
- causal guarantees;
- production deployment without pilot;
- universal performance across factories without local validation.
