# Enhanced Anomaly-Resistant Water Level Forecasting Model

## Overview
This document summarizes the key enhancements made to the anomaly-resistant LSTM model for water level forecasting. The model now combines robust anomaly detection with comprehensive feature engineering for improved prediction accuracy.

## Key Enhancements

### 1. Enhanced Feature Engineering

#### A. Lagged Features (`add_lagged_features`)
- **Purpose**: Provide historical water level context to predict peaks and valleys
- **Implementation**: Added lag features at intervals: 24h, 48h, 72h, 168h (1w), 336h (2w), 672h (4w)
- **Key Innovation**: Uses long lookbacks (up to 4 weeks) to escape potential anomalous periods
- **Data Source**: Uses `vst_raw_feature` (input version) to avoid look-ahead bias
- **Location**: `_3_lstm_model/feature_engineering.py`

#### B. Time-Based Features (`_add_time_features`)
- **Purpose**: Capture seasonal and cyclical patterns
- **Features Added**:
  - `month_sin`, `month_cos`: Monthly seasonal patterns
  - `day_of_year_sin`, `day_of_year_cos`: Annual seasonal patterns
- **Encoding**: Sin/cos encoding preserves cyclical nature

#### C. Cumulative Rainfall Features (`_add_cumulative_features`)
- **Purpose**: Capture rainfall accumulation effects on water levels
- **Features Added**:
  - `station_46_rain_1hour`: 1-hour cumulative rainfall
  - `station_46_rain_7hour`: 7-hour cumulative rainfall  
  - `station_46_rain_48hour`: 48-hour cumulative rainfall
  - `station_46_rain_90hour`: 90-hour cumulative rainfall

### 2. Enhanced Data Configuration

#### A. Flexible Training Modes
- **Quick Mode**: 5 years training (2018-2022), 1 year validation (2023)
- **Full Dataset Mode**: ~13 years training (2010-2022), 1 year validation (2023)
- **Standard Mode**: Moderate dataset for balanced training

#### B. Improved Data Splits
- More training data for better pattern learning
- Full year validation for comprehensive evaluation
- Maintains 2024 as test year for consistency

### 3. Anomaly Detection & Resistance

#### A. Perfect Anomaly Flags
- **Source**: Generated from known synthetic error locations
- **Integration**: Added as binary input feature to model
- **Purpose**: Train model to recognize and resist anomalous patterns

#### B. Memory Protection Mechanism
```python
# During anomalous periods, protect cell state memory
if is_anomalous.any():
    # Blend new cell state with last good state
    cell_state = last_good_cell_state * (1 - memory_decay) + new_cell_state * memory_decay
    hidden_state = new_hidden_state
else:
    # Update normally and store as last good state
    hidden_state = new_hidden_state
    cell_state = new_cell_state
    last_good_cell_state = cell_state.clone()
```

#### C. Weighted Loss Function
- **Normal periods**: Full weight (1.0) on prediction errors
- **Anomalous periods**: Reduced weight (0.3) to minimize corruption
- **Pattern penalty**: Additional penalty if model follows anomalous patterns

### 4. Model Architecture Enhancements

#### A. Dynamic Input Size
- Automatically adjusts to accommodate new features
- Handles feature engineering additions seamlessly

#### B. Debug Mode Optimization
- Ultra-lightweight debug (sampling every 100 timesteps)
- Only active in final training epoch
- Minimal performance impact during training

### 5. Configuration Updates

#### A. Enhanced Configuration (`alternating_config.py`)
```python
ALTERNATING_CONFIG = {
    # Feature Engineering
    'use_time_features': True,
    'use_cumulative_features': True, 
    'use_lagged_features': True,
    'lag_hours': [24, 48, 72, 168, 336, 672],  # 1d to 4w
    
    # Data Modes
    'quick_mode': False,
    'full_dataset_mode': False,
    
    # Anomaly Handling
    'use_anomaly_flags': True,
    'use_weighted_loss': True,
    'anomaly_weight': 0.3,
    'use_perfect_flags': True,
}
```

## Technical Implementation Details

### Feature Pipeline
1. **Load Raw Data**: Basic features (water level, temperature, rainfall)
2. **Add Time Features**: Seasonal encoding
3. **Add Cumulative Features**: Rainfall accumulation
4. **Add Lagged Features**: Historical water level context
5. **Add Anomaly Flags**: Perfect flags from error locations
6. **Scale Features**: Individual StandardScaler per feature (except binary flags)

### Training Process
1. **Data Preparation**: 5 years training, 1 year validation
2. **Model Initialization**: Dynamic input size based on final feature count
3. **Alternating Training**: 1 week original data, 1 week predictions
4. **Memory Protection**: Active during anomalous periods
5. **Weighted Loss**: Reduced impact from anomalous predictions

### Key Improvements Achieved

#### A. Better Peak/Valley Prediction
- **Before**: Model struggled with natural water level variations
- **After**: Accurate following of complex patterns and rapid changes
- **Reason**: Lagged features provide historical context

#### B. Robust Anomaly Resistance
- **Memory Protection**: Cell state preservation during anomalies
- **Perfect Detection**: Using known error locations for training
- **Weighted Training**: Reduced learning from anomalous periods

#### C. Enhanced Seasonal Understanding
- **Time Features**: Capture monthly and annual patterns
- **More Training Data**: 5 years vs 3 years for better pattern learning
- **Cumulative Features**: Rainfall effects on water levels

## File Structure
```
experiments/iterative_forecaster/
├── run_flagging_model.py           # Main execution script
├── alternating_trainer.py          # Enhanced trainer with flexible data modes
├── alternating_forecast_model.py   # Core model with memory protection
├── alternating_config.py           # Enhanced configuration
└── simple_anomaly_detector.py      # Perfect flag generation

_3_lstm_model/
└── feature_engineering.py          # Enhanced feature engineering
```

## Usage Example
```bash
# Quick mode with enhanced features
python run_flagging_model.py --experiment enhanced_model --quick_mode --seed 42

# Full dataset mode
python run_flagging_model.py --experiment full_enhanced --full_dataset --seed 42
```

## Results Summary
- **Prediction Accuracy**: Significantly improved peak/valley tracking
- **Anomaly Resistance**: Maintains predictions during corrupted periods
- **Training Efficiency**: Optimized debug mode for faster training
- **Feature Count**: ~18 features (3 base + 4 time + 4 rainfall + 6 lags + 1 anomaly flag) 