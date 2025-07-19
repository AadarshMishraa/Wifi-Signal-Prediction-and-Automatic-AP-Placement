import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

class WiFiDataCollector:
    def __init__(self, simulation_mode=True):
        """Initialize the WiFi data collector.
        
        Args:
            simulation_mode (bool): Whether to use simulated data
        """
        self.simulation_mode = simulation_mode
        
    def collect_training_data(self, duration_minutes=60, interval_seconds=1):
        """Collect WiFi signal strength data for training.
        
        Args:
            duration_minutes (int): Duration to collect data in minutes
            interval_seconds (int): Interval between measurements in seconds
            
        Returns:
            pd.DataFrame: Collected WiFi data
        """
        if self.simulation_mode:
            return self._generate_simulated_data(duration_minutes, interval_seconds)
        else:
            return self._collect_real_data(duration_minutes, interval_seconds)
    
    def _generate_simulated_data(self, duration_minutes, interval_seconds):
        """Generate simulated WiFi data.
        
        Args:
            duration_minutes (int): Duration to generate data for
            interval_seconds (int): Interval between measurements
            
        Returns:
            pd.DataFrame: Generated WiFi data
        """
        # Calculate number of samples
        n_samples = int((duration_minutes * 60) / interval_seconds)
        
        # Generate simulated access points
        ap_configs = [
            {'ssid': 'AP1', 'x': 0.2, 'y': 0.3, 'power': -30},
            {'ssid': 'AP2', 'x': 0.5, 'y': 0.4, 'power': -30},
            {'ssid': 'AP3', 'x': 0.8, 'y': 0.2, 'power': -30}
        ]
        
        # Generate data for each access point
        data = []
        for t in range(n_samples):
            timestamp = datetime.now().timestamp() + t * interval_seconds
            
            for ap in ap_configs:
                # Add random movement to simulate walking around
                x = np.random.normal(ap['x'], 0.1)
                y = np.random.normal(ap['y'], 0.1)
                
                # Calculate distance-based signal strength with noise
                distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
                rssi = ap['power'] - 20 * np.log10(max(distance, 0.1))
                rssi += np.random.normal(0, 2)  # Add noise
                
                data.append({
                    'timestamp': timestamp,
                    'ssid': ap['ssid'],
                    'bssid': f"00:11:22:33:44:{55+ap_configs.index(ap):02x}",
                    'rssi': rssi,
                    'channel': 1 + ap_configs.index(ap) * 5,
                    'security': 'WPA2',
                    'x': x,
                    'y': y
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        output_file = f'data/wifi_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Collected {len(df)} data points\n")
        
        return df
    
    def _collect_real_data(self, duration_minutes, interval_seconds):
        """Collect real WiFi data (not implemented).
        
        Args:
            duration_minutes (int): Duration to collect data
            interval_seconds (int): Interval between measurements
            
        Returns:
            pd.DataFrame: Collected WiFi data
        """
        raise NotImplementedError("Real data collection not implemented yet. Use simulation_mode=True")

if __name__ == "__main__":
    collector = WiFiDataCollector(simulation_mode=True)
    print("Starting WiFi data collection (simulation mode)...")
    data = collector.collect_training_data(duration_minutes=60)
    print(f"Collected {len(data)} data points")
    print("\nSample of collected data:")
    print(data.head())
