"""
Configuration for the Alternating Forecast Model.
"""

# Configuration for the alternating LSTM model
# Current: Experiment 3 (Time Features)
ALTERNATING_CONFIG = {
    # Model architecture
    'hidden_size': 128,         # Increased from 32 to 64
    'dropout': 0.25,
    
    # Training parameters
    'batch_size': (10*672)+672,# Batch size should always be at least 2 weeks, to allow for the periods
    'epochs': 50,              # Increased from 25 for better convergence
    'patience': 20,            # Increased from 15 - allow more time to improve
    'learning_rate': 0.001,
    # Forecasting parameters
    'week_steps': 672,         # Number of time steps in a week (15-min intervals: 4*24*7 = 672)
    'warmup_length': 672,
    # Anomaly detection parameters
    'threshold': 15.0,          # Increased from 5.0 - less sensitive to model prediction errors
    'window_size': 1500,        # Window size for MAD calculation
    
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
        #{
        #    'station_id': '21006845',
        #    'features': ['vst_raw', 'rainfall']
        #},
        #{
        #    'station_id': '21006847',
        #    'features': ['vst_raw', 'rainfall']
        #}
    ],
    # Feature engineering settings - EXPERIMENT 3: TIME FEATURES
    'use_time_features': False,        # ENABLED: month_sin, month_cos, day_of_year_sin, day_of_year_cos
    'use_cumulative_features': True, # FOR EXP 4: Enable cumulative rainfall features
    'use_lagged_features': False,     # Don't use lagged features
    
    # Do not include additional stations
    # The model will use only the primary station's features
}