"""
Configuration for the Alternating Forecast Model.
"""

# Configuration for the alternating LSTM model
ALTERNATING_CONFIG = {
    # Model architecture
    'hidden_size': 64,         # Increased from 32 to 64
    'num_layers': 2,           # Back to 2 layers for more capacity
    'dropout': 0.25,
    
    # Training parameters
    'batch_size': 64,          # Increased batch size
    'epochs': 5,              # More epochs for better convergence
    'patience': 8,
    'learning_rate': 0.001,
    
    # Quick mode for faster training with reduced data
    'quick_mode': True,       # When True, uses only 3 years training, 1 year validation
    
    # Forecasting parameters
    'week_steps': 672,         # Number of time steps in a week (15-min intervals: 4*24*7 = 672)
    
    # Anomaly detection parameters
    'threshold': 4.0,          # Threshold for detecting anomalies
    'window_size': 100,        # Window size for MAD calculation
    
    # Loss function
    'objective_function': 'smoothL1_loss',  # Using smoothL1 for robustness
    
    # Features
    'feature_cols': [
        'vst_raw_feature',     # Water level as input feature
        'temperature',         # Temperature
        'rainfall'             # Rainfall from primary station
    ],
    'output_features': ['vst_raw'],  # Target is water level
    
    # Empty feature_stations list for compatibility with DataPreprocessor
    'feature_stations': [],
    
    # Feature engineering settings
    'use_time_features': True,        # Enable time features (day of week, month, etc.)
    'use_cumulative_features': True,  # Enable cumulative features (e.g., cumulative rainfall)
    'use_lagged_features': False,     # Don't use lagged features
    
    # Do not include additional stations
    # The model will use only the primary station's features
} 