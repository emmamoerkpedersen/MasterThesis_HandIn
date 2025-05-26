"""
Configuration settings for error detection and imputation.
"""

# Error Detection Parameters
DETECTION_PARAMS = {
    'spike': {
        'window_size': 24,
        'threshold': 3.0
    },
    'gap': {
        'max_gap_hours': 1
    },
    'flatline': {
        'min_duration': 2,
        'max_duration': 48
    }
}

# Imputation Parameters
IMPUTATION_PARAMS = {
    'linear': {
        'max_gap': 24  # hours
    },
    'statistical': {
        'window_size': 168  # hours (1 week)
    }
}


# Synthetic Error Generation Parameters
SYNTHETIC_ERROR_PARAMS = {
    'spike': {
        'count_per_year': 1,  # Base count: 5 spikes per year
        'magnitude_range': (0.1, 1.0),  # Changed from (0.4, 0.8)
        'negative_positiv_ratio': 0.5,
        'recovery_time': 1
    },
    
     'drift': {
        'count_per_year': 1,  # Disabled by default
        'duration_range': [1168*2, 1168*3],  # Changed from [24, 168]
        'magnitude_range': [50, 150],   # Changed from [10, 50]
        'negative_positive_ratio': 0.5
    },
    
    'flatline': {
        'count_per_year': 0,  # Explicitly disabled
        'duration_range': (10, 200),
        'value_method': 'first_value'
    },
    
    'offset': {
        'count_per_year': 1,  # Base count: 2 offset periods per year
        'magnitude_range': (30, 700),  # Adjusted to account for the full range we want
        'negative_positiv_ratio': 0.5,
        'min_duration': 24,
        'max_duration': 1920  # 120 hours with 15-min timesteps
    },
    
    'noise': {
        'count_per_year': 1,  # Base count: 3 noise periods per year
        'duration_range': (168, 468),
        'intensity_range': (4, 10),  # Changed from (2, 4)
        'num_sub_segments_range': (15, 20), # Min and max number of sub-segments (steps)
        'segment_level_offset_range_abs': (50, 200), # Min/max absolute offset for each new level
        'segment_noise_std_abs': 1  # Base std for noise around each segment's new level
    },
    
    'baseline shift': {
        'count_per_year': 1,  # Disabled by default
        'magnitude_range': (100, 600),  # Changed from (200, 600)
        'negative_positive_ratio': 0
    },
    
    'missing_data': {
        'count_per_year': 0, # Explicitly disabled
        'min_length': 100,
        'max_length': 500
    },
    
    # Updated physical limits
    'PHYSICAL_LIMITS': {
        'min_value': 0,
        'max_value': 3000,
        'max_rate_of_change': 50
    }
}

# Physical Constraints
PHYSICAL_LIMITS = {
    'min_value': 0,  # Water level can't be negative
    'max_value': 4000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
}

# LSTM Configuration
LSTM_CONFIG = {
    'hidden_size': 128,         
    'num_layers': 2,                       
    'dropout': 0.25,  # N/A for standard model as we only have one layer             
    'batch_size': 1,
    'sequence_length': 70080,     
    'epochs': 50,
    'patience': 10,            
    'learning_rate': 0.001,    

    'warmup_length': 350,        # No warmup for standard model
        

    # 'smoothL1_loss', 'mse_loss', 'mae_loss
    'objective_function': 'smoothL1_loss',  # Changed to MSE for better stability
    'use_time_features': True,  
    'use_cumulative_features': True, 
    
    'feature_cols': [
        'rainfall', 'temperature'
    ],
    'output_features': ['vst_raw'],

    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['vst_raw', 'rainfall', 'temperature']
        },
        {
            'station_id': '21006847',
            'features': ['vst_raw', 'rainfall'] # temperature same as target station, so only included once
        }
    ]
}
