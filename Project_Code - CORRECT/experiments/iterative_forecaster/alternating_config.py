"""
Configuration for the Alternating Forecast Model.
"""

# Configuration for the alternating LSTM model
# Current: Experiment 3 (Time Features)
ALTERNATING_CONFIG = {
    # Model architecture
    'hidden_size': 24,         # Increased from 32 to 64
    'dropout': 0.25,
    
    # Training parameters
    'batch_size': (10*672)+672,# Batch size should always be at least 2 weeks, to allow for the periods
    'epochs': 40,              # More epochs for better convergence
    'patience': 5,
    'learning_rate': 0.001,
    # Forecasting parameters
    'warmup_length': 672,
    # Anomaly detection parameters
    'threshold': 50,          
    'window_size': 24, 

    'bce_weight': 1,          # Weight for anomaly detection loss
    'weight_factor': 5,       # Weight factor for class imbalance


    'week_steps': 672,
    # Quick mode for faster training with reduced data
    'quick_mode': False,       # When True, uses only 3 years training, 1 year validation

    # Loss function
    'objective_function': 'mae_loss',  # Using MAE lossx
    
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
    'use_time_features': True,        # ENABLED: month_sin, month_cos, day_of_year_sin, day_of_year_cos
    'use_cumulative_features': True, # FOR EXP 4: Enable cumulative rainfall features

    
    # Do not include additional stations
    # The model will use only the primary station's features
}