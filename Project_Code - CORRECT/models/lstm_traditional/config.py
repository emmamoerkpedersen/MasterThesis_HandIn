
# LSTM Model Configuration
LSTM_CONFIG = {
    # Model Architecture
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.25,
    'sequence_length': 70080,
    
    # Training Parameters
    'batch_size': 1,
    'epochs': 50,
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
