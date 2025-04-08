import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_scaler import FeatureScaler


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature columns list with base features
        self.feature_cols = config['feature_cols'].copy()
        
        # Add feature station columns dynamically
        for station in config['feature_stations']:
            for feature in station['features']:
                feature_name = f"feature_station_{station['station_id']}_{feature}"
                self.feature_cols.append(feature_name)
                
        self.output_features = config['output_features'][0]
        
        # Initialize the feature scaler
        self.feature_scaler = FeatureScaler(
            feature_cols=self.feature_cols,
            output_features=self.output_features,
            device=self.device
        )
    
    def load_and_split_data(self, project_root, station_id):
        """
        Load and Split data into features and target.
        """
        data_dir = project_root / "data_utils" / "Sample data"
        data = pd.read_pickle(data_dir / "preprocessed_data.pkl")

        # Check if station_id exists in the data dictionary, if not return empty dict
        station_data = data.get(station_id)
        if not station_data:
            raise ValueError(f"Station ID {station_id} not found in the data.")

        # Concatenate all station data columns
        df = pd.concat(station_data.values(), axis=1)

        # Add feature station data
        for station in self.config['feature_stations']:
            feature_station_id = station['station_id']
            feature_station_data = data.get(feature_station_id)
            
            if not feature_station_data:
                raise ValueError(f"Feature station ID {feature_station_id} not found in the data.")
            
            # Add each requested feature from the feature station
            for feature in station['features']:
                if feature not in feature_station_data:
                    raise ValueError(f"Feature {feature} not found in station {feature_station_id}")
                    
                feature_data = feature_station_data[feature][feature].rename(
                    f"feature_station_{feature_station_id}_{feature}"
                )
                df = pd.concat([df, feature_data], axis=1)

        # Start_date is first rainfall not nan, End_date is last vst_raw not nan
        start_date = pd.Timestamp('2010-01-04')
        end_date = pd.Timestamp('2025-01-07')
        # Cut dataframe
        data = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Fill temperature and rainfall Nan with bfill and ffill
        data.loc[:, 'temperature'] = data['temperature'].ffill().bfill()
        data.loc[:, 'rainfall'] = data['rainfall'].fillna(-1)
        data.loc[:, 'feature_station_21006845_vst_raw'] = data['feature_station_21006845_vst_raw'].fillna(-1)
        data.loc[:, 'feature_station_21006845_rainfall'] = data['feature_station_21006845_rainfall'].fillna(-1)
        data.loc[:, 'feature_station_21006847_vst_raw'] = data['feature_station_21006847_vst_raw'].fillna(-1)
        data.loc[:, 'feature_station_21006847_rainfall'] = data['feature_station_21006847_rainfall'].fillna(-1)
        print(f"  - Filled temperature and rainfall Nan with bfill and ffill")

        feature_cols = self.feature_cols
        target_feature = self.output_features
        all_features = list(set(feature_cols + [target_feature]))        
        #Filter data to only include the features and target feature
        data = data[all_features]

        # Split data based on years
        test_data = data[(data.index.year >= 2023) & (data.index.year <= 2024)]
        val_data = data[(data.index.year >= 2021) & (data.index.year <= 2022)]  # Validation is 2022-2023
        train_data = data[data.index.year < 2021]  # Training is everything before 2022
        
        print(f'Data shape: {data.shape}')
        print(f'Train data shape: {train_data.shape}')
        print(f'Val data shape: {val_data.shape}')
        print(f'Test data shape: {test_data.shape}')

        return train_data, val_data, test_data
    
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation. Scale data and create sequences.
        """
        # Get features and target
        feature_cols = self.feature_cols
        target_col = self.output_features   
        
        features = pd.concat([data[col] for col in feature_cols], axis=1)
        target = pd.DataFrame(data[target_col])

        # Scale data using FeatureScaler
        if is_training:
            scaled_features, scaled_target = self.feature_scaler.fit_transform(features, target)
        else:
            scaled_features, scaled_target = self.feature_scaler.transform(features, target)

        # Print feature ranges after scaling
        print("\nFeature ranges after scaling:")
        for i, col in enumerate(feature_cols):
            min_val = scaled_features[:, i].min()
            max_val = scaled_features[:, i].max()
            mean_val = scaled_features[:, i].mean()
            std_val = scaled_features[:, i].std()
            print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Print target range after scaling
        min_val = scaled_target.min()
        max_val = scaled_target.max()
        mean_val = scaled_target.mean()
        std_val = scaled_target.std()
        print(f"  {target_col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")

        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)
        print(f'X shape: {X.shape}, y shape: {y.shape}')
        
        # Convert to tensors and move to device
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def _create_sequences(self, features, targets):
         """
         Create sequences using the configured sequence length.
         """
         sequence_length = self.config.get('sequence_length', 5000)  # Get from config or default to 5000
         data_length = len(features)
         
         X, y = [], []
 
         # Create sequences based on configured sequence length
         for i in range(0, data_length, sequence_length):
             end_idx = min(i + sequence_length, data_length)
             feature_seq = features[i:end_idx]
             target_seq = targets[i:end_idx]
 
             # Pad sequences if needed
             if end_idx - i < sequence_length:
                 pad_length = sequence_length - (end_idx - i)
                 feature_seq = np.pad(feature_seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                 target_seq = np.pad(target_seq, (0, pad_length), mode='constant', constant_values=np.nan)
 
             X.append(feature_seq)
             y.append(target_seq)
 
         X = np.array(X)  # Shape: (num_sequences, sequence_length, num_features)
         y = np.array(y)[..., np.newaxis]  # Shape: (num_sequences, sequence_length, 1)
 
         return X, y
