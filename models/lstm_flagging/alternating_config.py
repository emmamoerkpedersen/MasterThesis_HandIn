"""
Configuration for the Alternating Forecast Model.
"""

# Configuration for the alternating LSTM model
# Current: Anomaly Flagging Approach
ALTERNATING_CONFIG = {
    # Model architecture
    'hidden_size': 24,
    'dropout': 0.25,
    
    # Training parameters
    'batch_size': (10*672)+672,# Batch size should always be at least 2 weeks, to allow for the periods
    'epochs': 45,              # TEMPORARILY REDUCED for quick testing (was 50)
    'patience': 10,
    'learning_rate': 0.001,
    # Forecasting parameters
    'warmup_length': 672,
    # Anomaly detection parameters
    'threshold': 5.0,          # For evaluation/visualization
    'window_size': 150,        # Window size for MAD calculation
    
    'week_steps': 672,
    # Quick mode for faster training with reduced data
    'quick_mode': True,       # When True, uses only 3 years training, 1 year validation
    
    # Loss function
    'objective_function': 'mae_loss',  # Using MAE loss
    
    # Features
    'feature_cols': [
        'vst_raw_feature',     # Water level as input feature
        'temperature',         # Temperature
        'rainfall'             # Rainfall from primary station
    ],
    'output_features': ['vst_raw'],  # Target is water level
    
    'feature_stations': [],
    
    # Feature engineering settings
    'use_time_features': True,        # ENABLED: month_sin, month_cos, day_of_year_sin, day_of_year_cos
    'use_cumulative_features': True, # Enable cumulative rainfall features

    # Data configuration
    'full_dataset_mode': False,      # When True, uses all available training data (2010-2022) and full validation year
    
    # NEW: Anomaly flagging approach
    'use_anomaly_flags': True,          # Enable anomaly flagging as input feature
    'use_weighted_loss': True,          # ENABLED: Anomaly aware loss (Experiment 2)
    'anomaly_weight': 0.3,              # Weight for anomalous periods (30% vs 100% for normal)
    'anomaly_detection_threshold': 3.0, # Z-score threshold for automatic detection
    'anomaly_detection_window': 100,   # Window size for MAD calculation
    'use_perfect_flags': True,          # Use perfect flags from known error locations

    # Flag column name
    'anomaly_flag_column': 'anomaly_flag',
}
