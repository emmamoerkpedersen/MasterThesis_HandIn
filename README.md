# Water Level Forecasting with LSTM Models

This project implements two different LSTM approaches for water level forecasting with anomaly detection capabilities.

## Models Overview

### 1. Traditional LSTM Model (`main_LSTM1_seq2seq.py`)
- **Type**: Standard sequence-to-sequence LSTM
- **Architecture**: Multi-layer LSTM with traditional training approach
- **Config**: `config.py` → `LSTM_CONFIG`
- **Location**: `models/lstm_traditional/`
- **Features**: Basic LSTM forecasting with configurable layers and hidden units

### 2. Flagging LSTM Model (`main_LSTM2_autoregressive.py`)
- **Type**: Alternating autoregressive LSTM with memory protection
- **Architecture**: LSTM with anomaly-aware training and memory protection mechanism
- **Config**: `models/lstm_flagging/alternating_config.py` → `ALTERNATING_CONFIG`
- **Location**: `models/lstm_flagging/`
- **Features**: Anomaly flags, weighted loss, memory protection during anomalous periods

## Running the Models

### Traditional LSTM
```bash
python main_LSTM1_seq2seq.py --station_id 21006846 --experiment my_experiment
```

### Flagging LSTM
```bash
python main_LSTM2_autoregressive.py --station_id 21006846 --experiment flagging_test --flag_method synthetic --anomaly_weight 0.3
```

**Key Arguments:**
- `--station_id`: Water level monitoring station (21006845, 21006846, 21006847)
- `--experiment`: Experiment name for results folder
- `--flag_method`: `synthetic` (from error injection) or `mad` (MAD outlier detection)
- `--anomaly_weight`: Weight for anomalous periods in loss function (0.1-1.0)
- `--quick_mode`: Use reduced dataset for faster testing
- `--error_multiplier`: Scale factor for synthetic error injection

## Project Structure

### Core Model Files
- `models/lstm_traditional/` - Traditional LSTM implementation
  - `model.py` - LSTMModel class
  - `train_model.py` - LSTM_Trainer class
- `models/lstm_flagging/` - Flagging LSTM implementation
  - `alternating_forecast_model.py` - AlternatingForecastModel class
  - `alternating_trainer.py` - AlternatingTrainer class
  - `alternating_config.py` - Model configuration
  - `simple_anomaly_detector.py` - Anomaly detection utilities

### Shared Components
- `shared/preprocessing/` - Data preprocessing and feature engineering
- `shared/diagnostics/` - Model evaluation and plotting utilities
- `shared/anomaly_detection/` - Anomaly detection algorithms (z-score, MAD)
- `shared/synthetic/` - Synthetic error generation for testing
- `shared/utils/` - General utilities and helper functions

### Data and Results
- `data_utils/Sample data/` - Preprocessed water level data
- `results/` - Model outputs and experiment results
  - `grid_search/` - Hyperparameter optimization results
  - `Iterative model results/` - Flagging model experiment results
  - `lstm/` - Traditional LSTM results

### Configuration
- `config.py` - Main configuration file with model parameters
- `models/lstm_flagging/alternating_config.py` - Flagging model specific config

## Key Features

### Traditional LSTM
- Multi-layer LSTM architecture
- Configurable sequence length and hidden units
- Standard backpropagation training
- Basic anomaly detection post-processing

### Flagging LSTM
- **Memory Protection**: Protects LSTM memory during anomalous periods
- **Weighted Loss**: Reduces learning from corrupted data
- **Anomaly Flags**: Binary inputs indicating data quality
- **Alternating Training**: Switches between ground truth and predictions
- **Synthetic Error Injection**: Offset, drift, spike, and noise errors

## Output Structure

Results are saved under `results/` with experiment-specific folders containing:
- `diagnostics/` - Model performance metrics and residual analysis
- `visualizations/` - Time series plots and model behavior analysis
- `anomaly_detection/` - Anomaly detection results and confidence analysis 