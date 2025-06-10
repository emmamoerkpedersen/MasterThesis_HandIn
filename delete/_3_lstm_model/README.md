# Water Level Anomaly Detection and Imputation Pipeline

This repository implements a modular pipeline for unsupervised anomaly detection and imputation in water level data from hydrometric stations using an LSTM model. The system is designed to both detect anomalies (with confidence estimates) and automatically impute values for high-confidence errors while flagging uncertain ones for manual review.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Flow](#pipeline-flow)
- [Key Components](#key-components)
  - [Data Preparation](#data-preparation)
  - [LSTM Model & Training](#lstm-model--training)
  - [Anomaly Detection](#anomaly-detection)
  - [Imputation](#imputation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Planned Extensions](#planned-extensions)
- [Usage](#usage)

---

## Overview

This project implements an end-to-end pipeline that:
- **Inputs preprocessed water level data** (prepared externally).
- **Trains an LSTM-based model** using transfer learning. The training incorporates a configurable **warmup period** at the start of each training cycle (e.g., at the beginning of each year) to help the model stabilize.
- **Detects anomalies** by combining the LSTM’s output with a set of specialized detectors (e.g., for spike, drift, and flatline errors). Each detector returns binary anomaly flags along with confidence scores.
- **Groups anomalies into segments** with defined start and end boundaries, and computes an aggregated confidence score for each segment.
- **Decides on imputation:**  
  - **High-Confidence segments:** Automatically impute new values using the model’s imputation mechanism.
  - **Low-Confidence segments:** Flag for manual review.
- **Tunes hyperparameters** using Optuna, allowing adjustments to the model architecture, training dynamics, and thresholds.

---

---

## Pipeline Flow

1. **Data Input:**  
   Use preprocessed water level data from the preprocessing module.

2. **LSTM Model Training:**  
   - The LSTM model is trained on the raw water level data.
   - A configurable **warmup period** is executed at the start of each training cycle (e.g., for the first few epochs of a yearly window) to allow the model to establish baseline patterns.
   - **Transfer Learning:** For each new training cycle (e.g., as new yearly data becomes available), the model is fine-tuned using weights carried over from previous training.

3. **Anomaly Detection:**  
   - The trained LSTM provides feature representations.
   - The `AnomalyDetector` module combines these features with specialized detectors that generate:
     - **Anomaly Flags:** Binary indicators per time step.
     - **Confidence Scores:** Representing the certainty of each detection.
   - Consecutive anomalous points are grouped into segments with start and end boundaries and an aggregated confidence score.

4. **Imputation & Review Decision:**  
   - The imputation module evaluates each anomaly segment:
     - **High Confidence:** Automatically imputes new values for the segment.
     - **Low Confidence:** Flags the segment for manual review.
   - Detailed reporting includes the start/end times for each anomaly segment and the corresponding confidence scores.

5. **Hyperparameter Tuning:**  
   - Hyperparameters (for model architecture, training, and detection thresholds) can be optimized using Optuna.

---

## Key Components

### LSTM Model & Training
- **File:** `lstm_model.py`
- **Details:**
  - Defines the LSTM model architecture.
  - Implements multiple training modes (e.g., sliding window, continuous, cumulative) with support for a warmup period.
  - Incorporates transfer learning by using previously learned weights for subsequent training cycles.

### Anomaly Detection
- **Files:**  
  - `detector.py` (main detector integration)  
  - `detectors/` (individual detector implementations)
- **Details:**  
  - Receives LSTM outputs and applies various statistical detectors.
  - Outputs anomaly flags along with confidence scores.
  - Segments consecutive anomalies and records start/end boundaries.

### Imputation
- **File:** `imputer.py`
- **Details:**  
  - Uses anomaly detection results to decide whether to impute or flag segments.
  - Applies automated imputation for high-confidence anomalies; low-confidence segments are flagged for manual review.

### Hyperparameter Tuning
- **File:** `hyperparameter_tuning.py`
- **Details:**  
  - Uses the Optuna framework to explore and optimize hyperparameters related to the model and data preparation.

---

## Planned Extensions

- **Detail Algorithm Implementations:**  
  Expand the placeholder sections for warmup logic, anomaly detection, and imputation with domain-specific algorithms.
- **Dynamic Threshold Adjustments:**  
  Introduce mechanisms for adapting thresholds based on evolving data characteristics.
- **Advanced Diagnostics & Reporting:**  
  Enhance reporting tools for better visualization of training progress, detection results, and imputation outcomes.
- **New Dataset Evaluation:**  
  Add support to run the pipeline on completely new datasets without synthetic anomaly injections.

---

## Usage

1. **Preparation:**  
   - Ensure that preprocessed water level data is available (produced by the `_1_preprocessing` module).
   - Configure parameters (e.g., warmup period, thresholds, model hyperparameters) in `config.py`.

2. **Running the Pipeline:**  
   - Execute `main.py` to run the full pipeline (training, detection, and imputation).
   - Use command-line flags (if implemented) to switch between evaluation modes (synthetic anomaly testing vs. new dataset evaluation).

3. **Review Outputs:**  
   - Check the generated diagnostics, logs, and reports to evaluate model performance and detect or impute anomalous segments.

---
