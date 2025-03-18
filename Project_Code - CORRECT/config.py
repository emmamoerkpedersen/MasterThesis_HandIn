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
        'frequency': 0.00003,
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
    
    'drift': {
        'frequency': 0.00003,
        # Wider range for drift durations
        'duration_range': [12, 168],  # Changed from [24, 168]
        # More varied magnitude range
        'magnitude_range': [5, 50],   # Changed from [10, 50]
        'negative_positive_ratio': 0.5,
        'context_aware': {
            'subtle_prob': 0.5,
            'medium_prob': 0.3,
            'obvious_prob': 0.2
        }
    },
    
    'flatline': {
        'frequency': 0.00003,
        # More varied durations
        'duration_range': (10, 200),  # Changed from (20, 200)
        'value_method': 'first_value'
    },
    
    'offset': {
        'frequency': 0.00003,
        # Wider range for offset magnitudes
        'magnitude_range': (20, 500),  # Changed from (50, 500)
        'negative_positiv_ratio': 0.7,
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
        'frequency': 0,  # Still disabled
        'duration_range': (4, 24),
        'intensity_range': (1, 4)  # Changed from (2, 4)
    },
    
    'baseline_shift': {
        'frequency': 0,  # Still disabled
        'magnitude_range': (100, 600),  # Changed from (200, 600)
        'negative_positive_ratio': 0.5
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

# LSTM Configuration with improved hyperparameters
LSTM_CONFIG = {
    'model_type': 'seq2seq_forecaster',
    'feature_cols': ['temperature', 'rainfall'],
    'output_features': ['vst_raw'],
    'hidden_size': 256,        # Keep as is for model capacity
    'num_layers': 1,           # Keep as is for model capacity
    'dropout': 0.15,          # Slightly reduced dropout for better fitting
    'batch_size': 1,          # Keep as is
    'learning_rate': 0.0005,   # Increased for faster learning
    'epochs': 3,             # Increased number of epochs
    'patience': 5,            # Decreased to be more selective
    'min_delta': 0.00001,     # More sensitive to improvements
    'max_chunk_size': 2500,   # Keep as is
    'smoothness_weight': 0.2,  # Reduced to allow more flexibility
    'attention_heads': 8,      # Keep as is
    'attention_dropout': 0.1,  # Keep as is
    
    # New parameters for target scaling
    'target_scaling': {
        'lower_padding': 0.2,  # 20% padding below min
        'upper_padding': 1.5   # 150% padding above max for better high value capture
    },
    
    # Enhanced loss function parameters
    'peak_detection_std': 1.5,  # More sensitive peak detection
    'peak_weight': 5.0,        # Increased importance of peaks
    'high_value_weight': 2.0,  # Additional weight for above-mean values
    'loss_weights': {
        'mse': 0.3,           # Base MSE component
        'peak': 0.4,          # Increased peak importance
        'gradient': 0.15,     # Gradient matching component
        'smoothness': 0.15    # Reduced smoothness constraint
    }
} 