import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class WiFiDataPreprocessor:
    def __init__(self):
        """Initialize the WiFi data preprocessor."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess(self, data):
        """Preprocess WiFi data for model training.
        
        Args:
            data (pd.DataFrame): Raw WiFi data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Encode categorical variables
        categorical_columns = ['ssid', 'bssid', 'security']
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        # Create signal quality metric
        df['signal_quality'] = (df['rssi'] + 100) / 70.0  # Normalize to 0-1 range
        
        # Calculate rolling statistics
        df['rssi_rolling_mean'] = df.groupby('ssid')['rssi'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['rssi_rolling_std'] = df.groupby('ssid')['rssi'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # Create channel interference feature
        df['channel_group'] = df['channel'] // 4  # Group nearby channels
        df['ap_count_per_channel'] = df.groupby('channel_group')['ssid'].transform('count')
        
        # Select and order features for model training
        feature_columns = [
            'rssi', 'signal_quality', 'channel',
            'hour', 'minute', 'day_of_week',
            'rssi_rolling_mean', 'rssi_rolling_std',
            'ap_count_per_channel'
        ]
        
        # Add encoded categorical columns
        feature_columns.extend([col + '_encoded' for col in categorical_columns])
        
        # Fill missing values
        df[feature_columns] = df[feature_columns].ffill().bfill()
        
        # Scale numerical features
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        # Add location information if available
        if 'x' in df.columns and 'y' in df.columns:
            df['distance_to_center'] = np.sqrt((df['x'] - 0.5)**2 + (df['y'] - 0.5)**2)
            feature_columns.extend(['x', 'y', 'distance_to_center'])
        
        return df[feature_columns]
