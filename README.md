# Water Level Forecasting with LSTM Models

This project implements two LSTM approaches for water level forecasting with anomaly detection capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages: `torch`, `pandas`, `numpy`, `matplotlib`, `scipy`

### Available Water Level Stations
- `21006845 (feature station for LSTM1)`, `21006846` (target), `21006847 (feature station for LSTM1)`

## 📖 Model Overview

| Model | Type | Key Features |
|-------|------|--------------|
| **Traditional LSTM** | Standard seq2seq | Multi-layer LSTM, basic anomaly detection |
| **Flagging LSTM** | Anomaly-aware | Memory protection, weighted loss, anomaly flags |

## 🏃 Running the Models

### 1. Traditional LSTM Model

**Basic usage:**
```bash
python main_LSTM1_seq2seq.py
```

**With options:**
```bash
# Train with synthetic errors and anomaly detection
python main_LSTM1_seq2seq.py --error_multiplier 1.5 --anomaly_detection --model_diagnostics

# Use test data for evaluation
python main_LSTM1_seq2seq.py --use_test_data --model_diagnostics
```

**Key Arguments:**
- `--error_multiplier FLOAT`: Scale factor for synthetic errors (default: 1.0, use `None` for no errors)
- `--anomaly_detection`: Enable anomaly detection analysis
- `--model_diagnostics`: Generate prediction plots and visualizations
- `--use_test_data`: Use test set for evaluation (default: validation set)
- `--advanced_diagnostics`: Generate detailed residual analysis

### 2. Flagging LSTM Model (Anomaly-Aware)

**Basic usage:**
```bash
python main_LSTM2_autoregressive.py
```

**Recommended usage:**
```bash
# Quick test with synthetic anomaly flags
python main_LSTM2_autoregressive.py --experiment quick_test --quick_mode

# Full experiment with custom anomaly weight
python main_LSTM2_autoregressive.py --experiment custom_weight --anomaly_weight 0.5 --station_id 21006846

# Use MAD-based anomaly detection
python main_LSTM2_autoregressive.py --flag_method mad --experiment mad_test
```

**Key Arguments:**
- `--station_id STR`: Station ID (default: '21006846')
- `--experiment STR`: Experiment name for results folder (default: 'flagging_test')
- `--flag_method STR`: Anomaly detection method - `synthetic` or `mad` (default: 'synthetic')
- `--anomaly_weight FLOAT`: Weight for anomalous periods in loss (0.1-1.0, default: 0.3)
- `--quick_mode`: Use reduced dataset for faster testing
- `--use_test_data`: Use test set for evaluation

## 📊 Output Structure

Results are automatically saved to:
```
results/
├── lstm/                           # Traditional LSTM results
└── Iterative model results/        # Flagging LSTM results
    └── experiment_{name}/
        ├── diagnostics/            # Performance metrics
        ├── visualizations/         # Time series plots
        └── anomaly_detection/      # Anomaly analysis
```

## 💡 Usage Examples

### Compare Both Models
```bash
# Run traditional LSTM
python main_LSTM1_seq2seq.py --model_diagnostics --anomaly_detection

# Run flagging LSTM with same configuration
python main_LSTM2_autoregressive.py --experiment comparison_test
```

### Anomaly Detection Focus
```bash
# Traditional model with synthetic errors
python main_LSTM1_seq2seq.py --error_multiplier 2.0 --anomaly_detection

# Flagging model with high anomaly awareness
python main_LSTM2_autoregressive.py --anomaly_weight 0.8 --experiment high_weight
```

### Quick Testing
```bash
# Fast traditional model run
python main_LSTM1_seq2seq.py --model_diagnostics

# Fast flagging model run
python main_LSTM2_autoregressive.py --quick_mode --experiment quick
```

## 🔧 Configuration

- **Traditional LSTM**: Edit `models/lstm_traditional/config.py`
- **Flagging LSTM**: Edit `models/lstm_flagging/alternating_config.py`
- **Synthetic Errors**: Edit `synthetic_error_config.py`

## 📁 Project Structure

```
├── main_LSTM1_seq2seq.py          # Traditional LSTM model
├── main_LSTM2_autoregressive.py   # Flagging LSTM model
├── models/
│   ├── lstm_traditional/          # Traditional LSTM implementation
│   └── lstm_flagging/             # Flagging LSTM implementation
├── shared/                        # Shared utilities
│   ├── anomaly_detection/         # Anomaly detection algorithms
│   ├── diagnostics/              # Model evaluation tools
│   └── synthetic/                # Synthetic error generation
├── data_utils/Sample data/        # Preprocessed data
└── results/                      # Model outputs
```

## 🎯 Key Features

### Traditional LSTM
- Standard sequence-to-sequence architecture
- Optional synthetic error injection
- Post-training anomaly detection
- Comprehensive diagnostic plots

### Flagging LSTM
- **Memory Protection**: LSTM state protection during anomalies
- **Weighted Loss**: Reduced learning from corrupted data
- **Anomaly Flags**: Binary indicators for data quality
- **Dual Detection**: Synthetic errors + MAD outlier detection 