"""
Configuration settings for error detection and imputation.
"""

# Physical Constraints
PHYSICAL_LIMITS = {
    'min_value': 0,  # Water level can't be negative
    'max_value': 3000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
}


# Synthetic Error Generation Parameters
SYNTHETIC_ERROR_PARAMS = {
    # Random seed for reproducible error generation
    'random_seed': 42,  # Set to None for non-deterministic behavior
    
    # Point-based errors (affect single points or very short periods)
    'spike': {
        'count_per_year': 0.5,
        'magnitude_range': (0.2, 1.0),  # Multiplier of current value
        'negative_positiv_ratio': 0.5,
        'recovery_time': 1  # 15-min intervals
    },
    
    # Period-based errors (affect longer periods)
    'offset': {
        'count_per_year': 0.25,
        'magnitude_range': (30, 700),  # Absolute value
        'duration_range': (24, 1920),  # 15-min intervals (6-120 hours)
        'negative_positiv_ratio': 0.5
    },
    
    'drift': {
        'count_per_year': 0.25,
        'magnitude_range': [50, 150],  # Absolute value
        'duration_range': [1168*2, 1168*3],  # 15-min intervals (24.3-36.5 days)
        'negative_positive_ratio': 0.5
    },
    
    'noise': {
        'count_per_year': 0.5,
        'magnitude_range': (50, 200),  # Absolute value for segment offsets
        'duration_range': (168, 468),  # 15-min intervals (42-117 hours)
        'negative_positive_ratio': 0.5,  # For segment direction
        'intensity_range': (4, 10),  # Multiplier for base noise
        'num_sub_segments_range': (15, 20),  # Number of steps in noise period
        'segment_noise_std_abs': 1  # Base standard deviation
    },

    
    # Physical limits for all error types
    'PHYSICAL_LIMITS': PHYSICAL_LIMITS
}

# LSTM Model Configuration
LSTM_CONFIG = {
    # Model Architecture
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.25,
    'sequence_length': 70080,
    
    # Training Parameters
    'batch_size': 1,
    'epochs': 1,
    'patience': 10,
    'learning_rate': 0.001,
    'warmup_length': 350,
    'objective_function': 'mse_loss',
    
    # Feature Configuration
    'use_time_features': True,
    'use_cumulative_features': True,
    'feature_cols': ['rainfall', 'temperature'],
    'output_features': ['vst_raw'],
    
    # Station Configuration
    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['vst_raw', 'rainfall', 'temperature']
        },
        {
            'station_id': '21006847',
            'features': ['vst_raw', 'rainfall']  # temperature same as target station
        }
    ]
}
