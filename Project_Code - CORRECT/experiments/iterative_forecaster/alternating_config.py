"""
Configuration for the Alternating Forecast Model.
"""

# Configuration for the alternating LSTM model
ALTERNATING_CONFIG = {
    # Model architecture
    'hidden_size': 128,         # Increased from 32 to 64
    'dropout': 0.25,
    
    # Training parameters
    'batch_size': (10*672)+672,# Batch size should always be at least 2 weeks, to allow for the periods
    'epochs': 50,              # More epochs for better convergence
    'patience': 15,
    'learning_rate': 0.001,
    # Forecasting parameters
    'week_steps': 672,         # Number of time steps in a week (15-min intervals: 4*24*7 = 672)
    'warmup_length': 672,
    # Anomaly detection parameters
    'threshold': 13.0,          # Threshold for detecting anomalies
    'window_size': 100,        # Window size for MAD calculation
    
    # Quick mode for faster training with reduced data
    'quick_mode': False,       # When True, uses only 3 years training, 1 year validation

    # Loss function
    'objective_function': 'mae_loss',  # Using MAE loss
    
    # Features
    'feature_cols': [
        'vst_raw_feature',     # Water level as input feature
        'temperature',         # Temperature
        'rainfall'             # Rainfall from primary station
    ],
    'output_features': ['vst_raw'],  # Target is water level
    

    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['vst_raw', 'rainfall']
        },
        {
            'station_id': '21006847',
            'features': ['vst_raw', 'rainfall']
        }
    ],
    # Feature engineering settings
    'use_time_features': True,        # Enable time features (day of week, month, etc.)
    'use_cumulative_features': True,  # Enable cumulative features (e.g., cumulative rainfall)
    'use_lagged_features': False,     # Don't use lagged features
    
    # Do not include additional stations
    # The model will use only the primary station's features
} 