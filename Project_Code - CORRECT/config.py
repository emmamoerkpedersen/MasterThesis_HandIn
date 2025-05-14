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
    # Add context-aware toggle
    'use_context_aware': False,
    
    'spike': {
        'count_per_year': 1,  # Base count: 5 spikes per year
        # Wider range to allow for more subtle spikes
        'magnitude_range': (0.1, 1.0),  # Changed from (0.4, 0.8)
        'negative_positiv_ratio': 0.5,
        'recovery_time': 1,
        # Add specific context-aware settings for spikes
        'context_aware': {
            'subtle_prob': 0.6,    # Probability of subtle spike
            'medium_prob': 0.3,    # Probability of medium spike
            'obvious_prob': 0.1,   # Probability of obvious spike
            'subtle_factor': 0.5,  # Multiplier for subtle range
            'medium_factor': 2.0,  # Multiplier for medium range
            'obvious_factor': 4.0  # Multiplier for obvious range
        }
    },
    
    # 'drift': {
    #     'count_per_year': 0,  # Disabled by default
    #     # Wider range for drift durations
    #     'duration_range': [500, 1168],  # Changed from [24, 168]
    #     # More varied magnitude range
    #     'magnitude_range': [5, 50],   # Changed from [10, 50]
    #     'negative_positive_ratio': 0.5,
    #     'context_aware': {
    #         'subtle_prob': 0.5,
    #         'medium_prob': 0.3,
    #         'obvious_prob': 0.2
    #     }
    # },
    
    # 'flatline': {
    #     'count_per_year': 0,  # Disabled by default
    #     # More varied durations
    #     'duration_range': (10, 200),  # Changed from (20, 200)
    #     'value_method': 'first_value'
    # },
    
    'offset': {
        'count_per_year': 1,  # Base count: 2 offset periods per year
        # Wider range for offset magnitudes
        'magnitude_range': (50, 500),  # Changed from (50, 500)
        'negative_positiv_ratio': 0.5,
        'min_duration': 24,
        'max_duration_multiplier': (15, 20),
        'magnitude_multiplier': (0.6, 1.4),  # Changed from (0.8, 1.2)
        'context_aware': {
            'subtle_prob': 0.5,
            'medium_prob': 0.3,
            'obvious_prob': 0.2
        }
    },
    
    'noise': {
        'count_per_year': 1,  # Base count: 3 noise periods per year
        'duration_range': (24, 168),
        'intensity_range': (4, 10)  # Changed from (2, 4)
    },
    
    # 'baseline_shift': {
    #     'count_per_year': 0,  # Disabled by default
    #     'magnitude_range': (100, 600),  # Changed from (200, 600)
    #     'negative_positive_ratio': 0.5
    # },
    
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
    'hidden_size': 32,         # Reduced from 64 to avoid overfitting
    'num_layers': 2,                       
    'dropout': 0.25,             
    'batch_size': 16,
    'sequence_length': 5000,     # Reduced sequence length to make processing more stable
    'epochs': 50,
    'patience': 5,            

    'warmup_length': 5,        # No warmup for standard model
    'learning_rate': 0.001,    

    # 'smoothL1_loss', 'mse_loss', ...
    'objective_function': 'mse_loss',  # Changed to MSE for better stability
    'use_time_features': True,  
    'use_cumulative_features': True, 
    # Add lag features for better prediction
    'use_lagged_features': False,
    'lag_hours': [72, 144, 288],  # Lag periods in hours
    
    'feature_cols': [
      #  'vst_raw_feature',
        'rainfall',
        'temperature',
        
    ],
    'output_features': ['vst_raw'],

    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['vst_raw', 'rainfall']
        },
        {
            'station_id': '21006847',
            'features': ['vst_raw', 'rainfall']
        }
    ]
}
