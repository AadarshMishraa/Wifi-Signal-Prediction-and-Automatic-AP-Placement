"""
Data Augmentation Utilities for WiFi ML
- Add realistic noise, interference, and fading
- Simulate environmental variability (materials, AP heights, obstacles)
"""
import numpy as np

def add_thermal_noise(rssi, noise_floor_dbm=-95, std_db=2.0):
    """Add Gaussian thermal noise to RSSI values."""
    noise = np.random.normal(0, std_db, size=np.shape(rssi))
    return rssi + noise

def add_interference(rssi, interference_dbm=-80, prob=0.1):
    """Randomly add interference spikes to RSSI values."""
    mask = np.random.rand(*np.shape(rssi)) < prob
    interference = np.zeros_like(rssi)
    interference[mask] = np.random.uniform(-10, 0, size=np.sum(mask))
    return rssi + interference

def add_fading(rssi, fading_type='rayleigh', K=5):
    """Add small-scale fading (Rayleigh or Rician) to RSSI values."""
    if fading_type == 'rayleigh':
        fading = np.random.rayleigh(scale=2, size=np.shape(rssi))
    elif fading_type == 'rician':
        fading = np.random.rayleigh(scale=2, size=np.shape(rssi)) + K
    else:
        fading = np.zeros_like(rssi)
    return rssi - fading  # Fading reduces RSSI

def simulate_environmental_variability(X, config=None):
    """Augment features to simulate different environments (materials, AP heights, obstacles)."""
    X_aug = X.copy()
    if 'ap_height' in X.dtype.names:
        X_aug['ap_height'] += np.random.uniform(-0.5, 0.5, size=X.shape[0])
    if 'material_id' in X.dtype.names:
        X_aug['material_id'] = np.random.choice([0,1,2,3], size=X.shape[0])
    if 'num_obstacles' in X.dtype.names:
        X_aug['num_obstacles'] += np.random.randint(-1, 2, size=X.shape[0])
    return X_aug 