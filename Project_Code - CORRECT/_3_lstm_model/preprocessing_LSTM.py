import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_scaler import FeatureScaler
from .feature_engineering import FeatureEngineer

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
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(config)
        
        # Initialize feature scaler
        self.feature_scaler = FeatureScaler(
            feature_cols=self.feature_cols,
            output_features=self.output_features,
            device=self.device
        )
        
        # Print feature configuration
        print(f"Feature configuration:")
        print(f"  - Base features: {self.feature_cols}")
        print(f"  - Use time features: {self.config.get('use_time_features', False)}")
        print(f"  - Use cumulative features: {self.config.get('use_cumulative_features', False)}")
    
    def update_feature_scaler(self):
        """
        Update the feature scaler with the latest feature columns.
        This should be called after adding new features.
        """
        self.feature_scaler = FeatureScaler(
            feature_cols=self.feature_cols,
            output_features=self.output_features,
            device=self.device
        )
        print(f"Updated feature scaler with columns: {self.feature_cols}")
    
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

        # Aggregate temperature to 30 days
        data.loc[:, 'temperature'] = data['temperature'].rolling(window=30, min_periods=1).mean()
        print(f"  - Aggregated temperature to 30 days")

        # Add cumulative rainfall features if enabled in config
        if self.config.get('use_cumulative_features', False):
            data = self._add_cumulative_features(data)
            # Update feature_cols with the new cumulative features
            self.feature_cols = self.feature_engineer.feature_cols.copy()
            print(f"  - Updated feature columns with cumulative features: {self.feature_cols}")
            # Update the feature scaler with the new feature columns
            self.update_feature_scaler()
        
        # Add time-based features if enabled in config
        if self.config.get('use_time_features', False):
            data = self._add_time_features(data)
            # Update feature_cols with the new time features
            self.feature_cols = self.feature_engineer.feature_cols.copy()
            print(f"  - Updated feature columns with time features: {self.feature_cols}")
            # Update the feature scaler with the new feature columns
            self.update_feature_scaler()


        feature_cols = self.feature_cols
        target_feature = self.output_features
        all_features = list(set(feature_cols + [target_feature]))        
        #Filter data to only include the features and target feature
        data = data[all_features]

        # Split data based on years
        test_data = data[(data.index.year >= 2022) & (data.index.year <= 2024)]
        val_data = data[(data.index.year >= 2021) & (data.index.year <= 2022)]  # Validation is 2022-2023
        train_data = data[data.index.year < 2021]  # Training is everything before 2022
        
        print(f"\nSplit Summary:")
        print(f"Training period: {train_data.index.min().year} - {train_data.index.max().year}")
        print(f"Validation period: {val_data.index.min().year} - {val_data.index.max().year}")
        print(f"Test year: {test_data.index.min().year}")
        
        print(f'\nData shapes:')
        print(f'Total data: {data.shape}')
        print(f'Train data: {train_data.shape}')
        print(f'Validation data: {val_data.shape}')
        print(f'Test data: {test_data.shape}')

        return train_data, val_data, test_data
    
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation. Scale data and create sequences.
        """
        # Get features and target
        feature_cols = self.feature_cols
        target_col = self.output_features   
        
        # Ensure we're using the most up-to-date feature columns
        if self.config.get('use_time_features', False) or self.config.get('use_cumulative_features', False):
            # Update feature_cols with the latest from feature_engineer
            feature_cols = self.feature_engineer.feature_cols.copy()
            print(f"Using updated feature columns in prepare_data: {feature_cols}")
        
        features = pd.concat([data[col] for col in feature_cols], axis=1)
        target = pd.DataFrame(data[target_col])
        
        # Print diagnostic information about NaN values in target
        nan_count = target[target_col].isna().sum()
        total_count = len(target)
        print(f"\nTarget NaN diagnostics before scaling:")
        print(f"  {target_col}: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.2f}%)")
        
        if nan_count == total_count:
            print("  WARNING: All target values are NaN!")
            # Fill NaN values with a default value (e.g., 0) to allow scaling to proceed
            target[target_col] = target[target_col].fillna(0)
            print("  Filled NaN values with 0 to allow scaling to proceed")

        # Scale data using FeatureScaler
        if is_training:
            scaled_features, scaled_target = self.feature_scaler.fit_transform(features, target)
        else:
            scaled_features, scaled_target = self.feature_scaler.transform(features, target)
            
        # Print diagnostic information about NaN values in scaled target
        nan_count_scaled = np.isnan(scaled_target).sum()
        total_count_scaled = len(scaled_target)
        print(f"\nTarget NaN diagnostics after scaling:")
        print(f"  {target_col}: {nan_count_scaled}/{total_count_scaled} NaN values ({nan_count_scaled/total_count_scaled*100:.2f}%)")
        
        if nan_count_scaled == total_count_scaled:
            print("  WARNING: All scaled target values are NaN!")
            # Fill NaN values with a default value (e.g., 0) to allow model training to proceed
            scaled_target = np.nan_to_num(scaled_target, nan=0)
            print("  Filled NaN values with 0 to allow model training to proceed")

        # Print feature ranges after scaling
        print("\nFeature ranges after scaling:")
        for i, col in enumerate(feature_cols):
            min_val = scaled_features[:, i].min()
            max_val = scaled_features[:, i].max()
            mean_val = scaled_features[:, i].mean()
            std_val = scaled_features[:, i].std()
            print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Print target range after scaling
        # Use NaN-aware functions to handle the few NaN values
        min_val = np.nanmin(scaled_target)
        max_val = np.nanmax(scaled_target)
        mean_val = np.nanmean(scaled_target)
        std_val = np.nanstd(scaled_target)
        print(f"Target range after scaling:")
        print(f"  {target_col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
    
        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)

        # Only print basic shape info for debugging
        print(f"{'Training' if is_training else 'Validation'} data: {X.shape[0]} sequences of length {X.shape[1]}")
        
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

    def _add_time_features(self, data):
        """
        Add time-based features to the data.
        """
        return self.feature_engineer._add_time_features(data)
        
    def _add_cumulative_features(self, data):
        """
        Add cumulative features to the data.
        """
        return self.feature_engineer._add_cumulative_features(data)
