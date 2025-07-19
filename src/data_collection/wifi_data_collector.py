"""Module for collecting and simulating WiFi signal strength data."""

import numpy as np
from typing import List, Tuple, Optional
from src.physics.materials import SignalPath, Material, MATERIALS

class WiFiDataCollector:
    """Collects and simulates WiFi signal strength data with material effects."""
    
    def __init__(self, tx_power: float = 20.0, frequency: float = 2.4e9):
        """Initialize the WiFi data collector.
        
        Args:
            tx_power (float): Transmit power in dBm
            frequency (float): Signal frequency in Hz (default: 2.4 GHz)
        """
        self.tx_power = tx_power
        self.frequency = frequency
        self.noise_floor = -96.0  # Typical WiFi noise floor in dBm
        
    def calculate_free_space_loss(self, distance: float) -> float:
        """Calculate free space path loss.
        
        Args:
            distance (float): Distance in meters
            
        Returns:
            float: Path loss in dB
        """
        c = 3e8  # Speed of light
        wavelength = c / self.frequency
        
        # Free space path loss formula
        if distance == 0:
            return 0
        return 20 * np.log10(4 * np.pi * distance / wavelength)
        
    def calculate_material_loss(self, signal_path: SignalPath) -> float:
        """Calculate signal loss due to materials.
        
        Args:
            signal_path (SignalPath): Path containing material layers
            
        Returns:
            float: Material loss in dB
        """
        return signal_path.calculate_total_attenuation(self.frequency)
        
    def add_multipath_effects(self, rssi: float, n_paths: int = 3) -> float:
        """Simulate multipath effects on signal strength.
        
        Args:
            rssi (float): Original RSSI value
            n_paths (int): Number of reflection paths to simulate
            
        Returns:
            float: RSSI with multipath effects
        """
        # Generate random path delays and attenuations
        path_losses = np.random.uniform(3, 20, n_paths)  # Additional loss per path in dB
        path_phases = np.random.uniform(0, 2*np.pi, n_paths)  # Random phases
        
        # Convert RSSI to linear power (handle negative values)
        power_linear = 10 ** (rssi/10) if rssi > -100 else 1e-10
        
        # Add multipath components
        for loss, phase in zip(path_losses, path_phases):
            reflected_power = 10 ** ((rssi - loss)/10) if (rssi - loss) > -100 else 1e-10
            power_linear += reflected_power * np.cos(phase)  # Coherent addition
        
        # Ensure power is positive before log
        power_linear = max(power_linear, 1e-10)
            
        # Convert back to dB
        return 10 * np.log10(power_linear)
        
    def calculate_rssi(self, 
                      distance: float, 
                      signal_path: Optional[SignalPath] = None,
                      include_multipath: bool = True) -> float:
        """Calculate RSSI at a given distance considering materials and multipath.
        
        Args:
            distance (float): Distance in meters
            signal_path (Optional[SignalPath]): Path with materials
            include_multipath (bool): Whether to include multipath effects
            
        Returns:
            float: RSSI value in dBm
        """
        # Calculate free space path loss
        path_loss = self.calculate_free_space_loss(distance)
        
        # Add material losses if path is provided
        material_loss = 0
        if signal_path is not None:
            material_loss = self.calculate_material_loss(signal_path)
            
        # Calculate basic RSSI
        rssi = self.tx_power - path_loss - material_loss
        
        # Add multipath effects if requested
        if include_multipath:
            rssi = self.add_multipath_effects(rssi)
            
        # Ensure we don't go below noise floor
        return max(rssi, self.noise_floor)
        
    def collect_samples(self, 
                       points: List[Tuple[float, float]], 
                       ap_location: Tuple[float, float],
                       materials_grid: Optional[List[List[Material]]] = None) -> np.ndarray:
        """Collect RSSI samples for given points considering materials.
        
        Args:
            points: List of (x, y) measurement points
            ap_location: (x, y) location of access point
            materials_grid: Optional 2D grid of materials
            
        Returns:
            numpy array of RSSI values
        """
        samples = []
        ap_x, ap_y = ap_location
        
        for x, y in points:
            # Calculate distance
            distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
            
            # Create signal path if materials grid is provided
            signal_path = None
            if materials_grid is not None:
                signal_path = SignalPath()
                
                # Simple ray tracing - check materials along direct line
                if distance > 0.1:  # Only do ray tracing for non-zero distances
                    # Calculate step sizes for ray tracing
                    steps = max(int(distance * 2), 3)  # At least 3 steps to avoid division by zero
                    
                    # Safety check to prevent division by zero
                    if steps > 1:
                        dx = (x - ap_x) / (steps - 1)
                        dy = (y - ap_y) / (steps - 1)
                        
                        # Track unique materials encountered
                        materials_seen = set()
                        
                        # Trace ray from AP to measurement point
                        for i in range(steps):
                            # Current position along ray
                            curr_x = ap_x + dx * i
                            curr_y = ap_y + dy * i
                            
                            # Convert to grid indices
                            grid_x = int(curr_x * 2)  # Assuming 0.5m resolution
                            grid_y = int(curr_y * 2)
                            
                            # Check if indices are valid
                            if (0 <= grid_y < len(materials_grid) and 
                                0 <= grid_x < len(materials_grid[0])):
                                material = materials_grid[grid_y][grid_x]
                                if isinstance(material, Material) and material not in materials_seen:
                                    materials_seen.add(material)
                                    signal_path.add_layer(material)
            
            # Calculate RSSI
            rssi = self.calculate_rssi(distance, signal_path)
            samples.append(rssi)
            
        return np.array(samples)
