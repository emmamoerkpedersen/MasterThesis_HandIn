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
    # Point-based errors (affect single points or very short periods)
    'spike': {
        'count_per_year': 1,
        'magnitude_range': (0.1, 1.0),  # Multiplier of current value
        'negative_positiv_ratio': 0.5,
        'recovery_time': 1  # 15-min intervals
    },
    
    'baseline shift': {
        'count_per_year': 1,
        'magnitude_range': (100, 600),  # Absolute value
        'negative_positive_ratio': 0
    },
    
    # Period-based errors (affect longer periods)
    'offset': {
        'count_per_year': 1,
        'magnitude_range': (30, 700),  # Absolute value
        'duration_range': (24, 1920),  # 15-min intervals (6-120 hours)
        'negative_positiv_ratio': 0.5
    },
    
    'drift': {
        'count_per_year': 1,
        'magnitude_range': [50, 150],  # Absolute value
        'duration_range': [1168*2, 1168*3],  # 15-min intervals (24.3-36.5 days)
        'negative_positive_ratio': 0.5
    },
    
    'noise': {
        'count_per_year': 1,
        'magnitude_range': (50, 200),  # Absolute value for segment offsets
        'duration_range': (168, 468),  # 15-min intervals (42-117 hours)
        'negative_positive_ratio': 0.5,  # For segment direction
        'intensity_range': (4, 10),  # Multiplier for base noise
        'num_sub_segments_range': (15, 20),  # Number of steps in noise period
        'segment_noise_std_abs': 1  # Base standard deviation
    },
    
    # Disabled error types
    'flatline': {
        'count_per_year': 0,
        'magnitude_range': (0, 0),  # Not applicable for flatline
        'duration_range': (10, 200),
        'negative_positive_ratio': 0.5,
        'value_method': 'first_value'
    },
    
    'missing_data': {
        'count_per_year': 0,
        'magnitude_range': (0, 0),  # Not applicable for missing data
        'min_length': 100,
        'max_length': 500
    },
    
    # Physical limits for all error types
    'PHYSICAL_LIMITS': PHYSICAL_LIMITS
}

# LSTM Model Configuration
LSTM_CONFIG = {
    # Model Architecture
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.25,
    'sequence_length': 70080,
    
    # Training Parameters
    'batch_size': 1,
    'epochs': 50,
    'patience': 10,
    'learning_rate': 0.001,
    'warmup_length': 350,
    'objective_function': 'smoothL1_loss',
    
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
