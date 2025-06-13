# Water Level Forecasting with LSTM Models

This project implements two LSTM approaches for water level forecasting with anomaly detection capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages: `torch`, `pandas`, `numpy`, `matplotlib`, `scipy`
  
- Run data_preprocessing/Preprocessing_data.py to get the relevant .pkl file for running models
- 
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

**With options (examples):**
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

**Recommended usage (examples):**
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

## 🔧 Synthetic Error Framework

This project includes a comprehensive synthetic error injection system to test model robustness against common water level sensor issues.

### Error Types

| Error Type | Description | Example Use Case |
|------------|-------------|------------------|
| **Spike** | Single-point anomalies | Sensor glitches, electrical interference |
| **Offset** | Sudden level shifts | Calibration drift, sensor repositioning |
| **Drift** | Gradual changes over time | Sensor degradation, environmental factors |
| **Noise** | Periods of increased variance | Communication issues, environmental interference |

### Configuration

Edit `synthetic_error_config.py` to control error generation:

```python
SYNTHETIC_ERROR_PARAMS = {
    'spike': {
        'count_per_year': 0.5,        # Number of spikes per year
        'magnitude_range': (0.2, 1.0), # Spike intensity
    },
    'offset': {
        'count_per_year': 0.25,       # Number of offsets per year
        'magnitude_range': (30, 700),  # Offset size (mm)
        'duration_range': (24, 1920),  # Duration (15-min intervals)
    },
    # ... more error types
}
```

### How It Works

1. **Automatic Scaling**: Errors are distributed based on dataset length (years × count_per_year)
2. **Non-Overlapping**: Built-in collision detection prevents error overlap
3. **Physical Constraints**: Maintains realistic water level limits (0-3000mm)
4. **Reproducible**: Configurable random seeds for consistent experiments

### Usage Examples

```bash
# No synthetic errors (clean data)
python main_LSTM1_seq2seq.py

# Light error injection (1x base rates)
python main_LSTM1_seq2seq.py --error_multiplier 1.0

# Heavy error injection (2x base rates)
python main_LSTM1_seq2seq.py --error_multiplier 2.0
```

## ⚙️ Configuration Files

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
