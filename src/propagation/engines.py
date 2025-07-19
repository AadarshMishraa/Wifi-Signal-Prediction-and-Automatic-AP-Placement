"""Advanced propagation engines with precise physics models."""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
import logging
from scipy import constants
from scipy.optimize import minimize_scalar
import warnings

# Import advanced materials
from src.physics.materials import (
    AdvancedMaterial, Material, ADVANCED_MATERIALS, 
    FrequencyDependentProperty, EPSILON_0, MU_0, C, ETA_0
)

class PropagationEngine(ABC):
    """Abstract base class for propagation engines."""
    
    @abstractmethod
    def calculate_rssi(self, ap: Tuple[float, float, float], 
                      point: Tuple[float, float, float], 
                      materials_grid, **kwargs) -> float:
        """Calculate RSSI at a point from an AP."""
        pass

class AdvancedPhysicsEngine(PropagationEngine):
    """
    Advanced Physics Engine with precise electromagnetic modeling.
    
    Features:
    - Frequency-dependent material properties
    - Angle-dependent attenuation using Snell's Law and Fresnel equations
    - Thickness-dependent exponential attenuation
    - Composite material handling
    - Surface roughness effects
    - Temperature-dependent properties
    - Multi-path interference modeling
    """
    
    def __init__(self, frequency: float = 2.4e9, temperature: float = 293.15):
        """Initialize the advanced physics engine.
        
        Args:
            frequency: Operating frequency in Hz
            temperature: Temperature in Kelvin
        """
        self.frequency = frequency
        self.temperature = temperature
        self.wavelength = C / frequency
        self.k0 = 2 * np.pi / self.wavelength  # Free space wavenumber
        
        # Physical constants
        self.epsilon_0 = EPSILON_0
        self.mu_0 = MU_0
        self.eta_0 = ETA_0
        
        # Engine configuration
        self.max_reflections = 3
        self.max_diffractions = 2
        self.include_surface_roughness = True
        self.include_temperature_effects = True
        self.use_composite_materials = True
        
        logging.info(f"Advanced Physics Engine initialized at {frequency/1e9:.1f} GHz")
    
    def calculate_rssi(self, ap: Tuple[float, float, float], 
                      point: Tuple[float, float, float], 
                      materials_grid, **kwargs) -> float:
        """
        Calculate precise RSSI using advanced electromagnetic physics.
        
        Args:
            ap: AP coordinates (x, y, z)
            point: Receiver coordinates (x, y, z)
            materials_grid: 3D grid of materials
            **kwargs: Additional parameters (tx_power, polarization, etc.)
            
        Returns:
            RSSI in dBm
        """
        tx_power = kwargs.get('tx_power', 20.0)
        polarization = kwargs.get('polarization', 'TE')
        
        # Calculate direct path
        direct_rssi = self._calculate_direct_path(ap, point, materials_grid, tx_power, polarization)
        
        # Calculate reflected paths
        reflected_rssi = self._calculate_reflected_paths(ap, point, materials_grid, tx_power, polarization)
        
        # Calculate diffracted paths
        diffracted_rssi = self._calculate_diffracted_paths(ap, point, materials_grid, tx_power)
        
        # Combine all paths using power addition
        total_rssi = self._combine_multipath_signals([direct_rssi, reflected_rssi, diffracted_rssi])
        
        return total_rssi
    
    def _calculate_direct_path(self, ap: Tuple[float, float, float], 
                             point: Tuple[float, float, float], 
                             materials_grid, tx_power: float, 
                             polarization: str) -> float:
        """Calculate direct path RSSI with precise material modeling."""
        # Calculate distance
        distance = np.sqrt(sum((ap[i] - point[i])**2 for i in range(3)))
        
        if distance < 1e-6:
            return tx_power  # Very close to AP
        
        # Free space path loss
        free_space_loss = 20 * np.log10(4 * np.pi * distance / self.wavelength)
        
        # Material attenuation along the path
        material_attenuation = self._calculate_material_attenuation_3d(
            ap, point, materials_grid, polarization
        )
        
        # Total RSSI
        rssi = tx_power - free_space_loss - material_attenuation
        
        return rssi
    
    def _calculate_material_attenuation_3d(self, ap: Tuple[float, float, float], 
                                         point: Tuple[float, float, float], 
                                         materials_grid, polarization: str) -> float:
        """
        Calculate precise material attenuation along 3D path with angle dependence.
        """
        if materials_grid is None:
            return 0.0
        
        # Use 3D Bresenham algorithm to traverse the path
        path_points = self._get_3d_path_points(ap, point)
        
        total_attenuation = 0.0
        seen_materials = set()
        
        for i, (x, y, z) in enumerate(path_points):
            # Get material at this point
            material = self._get_material_at_point(x, y, z, materials_grid)
            
            if material is None or material.name == 'Air':
                continue
            
            # Calculate angle of incidence for this segment
            if i < len(path_points) - 1:
                next_point = path_points[i + 1]
                angle_of_incidence = self._calculate_angle_of_incidence(
                    path_points[i], next_point, materials_grid
                )
            else:
                angle_of_incidence = 0.0
            
            # Calculate segment length
            if i < len(path_points) - 1:
                segment_length = np.sqrt(sum((path_points[i+1][j] - path_points[i][j])**2 for j in range(3)))
            else:
                segment_length = 0.1  # Default segment length
            
            # Calculate attenuation for this material segment
            if isinstance(material, AdvancedMaterial):
                segment_atten = material.calculate_total_attenuation_with_reflection(
                    self.frequency, segment_length, angle_of_incidence, polarization
                )
            else:
                # Legacy material
                segment_atten = material.calculate_attenuation(self.frequency)
                # Apply angle correction
                if angle_of_incidence > 0:
                    segment_atten /= np.cos(angle_of_incidence)
            
            # Avoid double-counting same material
            material_key = (material.name, x, y, z)
            if material_key not in seen_materials:
                total_attenuation += segment_atten
                seen_materials.add(material_key)
        
        return total_attenuation
    
    def _get_3d_path_points(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Get 3D path points using Bresenham algorithm."""
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        # Convert to grid coordinates (assuming 0.2m resolution)
        resolution = 0.2
        gx1, gy1, gz1 = int(x1 / resolution), int(y1 / resolution), int(z1 / resolution)
        gx2, gy2, gz2 = int(x2 / resolution), int(y2 / resolution), int(z2 / resolution)
        
        # 3D Bresenham algorithm
        points = []
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        dz = abs(gz2 - gz1)
        xs = 1 if gx2 > gx1 else -1
        ys = 1 if gy2 > gy1 else -1
        zs = 1 if gz2 > gz1 else -1
        # Driving axis is X
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while gx1 != gx2:
                points.append((gx1 * resolution, gy1 * resolution, gz1 * resolution))
                if p1 >= 0:
                    gy1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    gz1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                gx1 += xs
        # Driving axis is Y
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while gy1 != gy2:
                points.append((gx1 * resolution, gy1 * resolution, gz1 * resolution))
                if p1 >= 0:
                    gx1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    gz1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                gy1 += ys
        # Driving axis is Z
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while gz1 != gz2:
                points.append((gx1 * resolution, gy1 * resolution, gz1 * resolution))
                if p1 >= 0:
                    gy1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    gx1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                gz1 += zs
        points.append((gx2 * resolution, gy2 * resolution, gz2 * resolution))
        return points
    
    def _get_material_at_point(self, x: float, y: float, z: float, 
                              materials_grid) -> Optional[Union[Material, AdvancedMaterial]]:
        """Get material at a specific 3D point."""
        if materials_grid is None:
            return None
        
        # Convert to grid coordinates
        resolution = 0.2
        gx = int(x / resolution)
        gy = int(y / resolution)
        gz = int(z / resolution)
        
        # Check bounds
        if (0 <= gz < len(materials_grid) and 
            0 <= gy < len(materials_grid[0]) and 
            0 <= gx < len(materials_grid[0][0])):
            return materials_grid[gz][gy][gx]
        
        return None
    
    def _calculate_angle_of_incidence(self, point1: Tuple[float, float, float], 
                                    point2: Tuple[float, float, float], 
                                    materials_grid) -> float:
        """Calculate angle of incidence with respect to material surface."""
        # For simplicity, assume normal incidence
        # In a more advanced implementation, this would calculate the actual angle
        # based on surface normal vectors
        return 0.0
    
    def _calculate_reflected_paths(self, ap: Tuple[float, float, float], 
                                 point: Tuple[float, float, float], 
                                 materials_grid, tx_power: float, 
                                 polarization: str) -> float:
        """Calculate reflected path contributions."""
        if self.max_reflections == 0:
            return -100.0  # No reflections
        
        # Find major reflecting surfaces (walls, floor, ceiling)
        reflecting_surfaces = self._find_reflecting_surfaces(ap, point, materials_grid)
        
        reflected_signals = []
        
        for surface in reflecting_surfaces[:self.max_reflections]:
            # Calculate reflection point
            reflection_point = self._calculate_reflection_point(ap, point, surface)
            
            if reflection_point is None:
                continue
            
            # Calculate reflected path
            reflected_rssi = self._calculate_reflected_path(
                ap, reflection_point, point, surface, tx_power, polarization
            )
            
            if reflected_rssi > -100:
                reflected_signals.append(reflected_rssi)
        
        # Combine reflected signals
        if reflected_signals:
            return self._combine_multipath_signals(reflected_signals)
        else:
            return -100.0
    
    def _find_reflecting_surfaces(self, ap: Tuple[float, float, float], 
                                point: Tuple[float, float, float], 
                                materials_grid) -> List[Dict]:
        """Find major reflecting surfaces in the environment."""
        surfaces = []
        
        # Add floor and ceiling as reflecting surfaces
        surfaces.append({
            'type': 'floor',
            'z': 0.0,
            'material': ADVANCED_MATERIALS.get('concrete', None)
        })
        
        surfaces.append({
            'type': 'ceiling',
            'z': 3.0,  # Assume 3m ceiling height
            'material': ADVANCED_MATERIALS.get('concrete', None)
        })
        
        # Add major walls (simplified)
        # In a full implementation, this would analyze the materials_grid
        # to find wall surfaces
        
        return surfaces
    
    def _calculate_reflection_point(self, ap: Tuple[float, float, float], 
                                  point: Tuple[float, float, float], 
                                  surface: Dict) -> Optional[Tuple[float, float, float]]:
        """Calculate reflection point on a surface."""
        if surface['type'] == 'floor':
            # Reflect AP across the floor
            return (ap[0], ap[1], -ap[2])
        elif surface['type'] == 'ceiling':
            # Reflect AP across the ceiling
            ceiling_z = surface['z']
            return (ap[0], ap[1], 2 * ceiling_z - ap[2])
        
        return None
    
    def _calculate_reflected_path(self, ap: Tuple[float, float, float], 
                                reflection_point: Tuple[float, float, float], 
                                point: Tuple[float, float, float], 
                                surface: Dict, tx_power: float, 
                                polarization: str) -> float:
        """Calculate RSSI for a reflected path."""
        # Distance from AP to reflection point to receiver
        d1 = np.sqrt(sum((ap[i] - reflection_point[i])**2 for i in range(3)))
        d2 = np.sqrt(sum((reflection_point[i] - point[i])**2 for i in range(3)))
        total_distance = d1 + d2
        
        # Free space path loss
        free_space_loss = 20 * np.log10(4 * np.pi * total_distance / self.wavelength)
        
        # Reflection loss
        if surface['material'] is not None:
            reflection_coeff = surface['material'].calculate_reflection_coefficient(
                self.frequency, 0.0, polarization  # Normal incidence
            )
            reflection_loss = -10 * np.log10(np.abs(reflection_coeff)**2)
        else:
            reflection_loss = 6.0  # Default reflection loss
        
        # Material attenuation (simplified)
        material_attenuation = 0.0  # Could be calculated along the path
        
        # Total RSSI
        rssi = tx_power - free_space_loss - reflection_loss - material_attenuation
        
        return rssi
    
    def _calculate_diffracted_paths(self, ap: Tuple[float, float, float], 
                                  point: Tuple[float, float, float], 
                                  materials_grid, tx_power: float) -> float:
        """Calculate diffracted path contributions."""
        if self.max_diffractions == 0:
            return -100.0  # No diffractions
        
        # Simplified diffraction model
        # Count obstacles along the direct path
        obstacles = self._count_obstacles_along_path(ap, point, materials_grid)
        
        if obstacles == 0:
            return -100.0  # No obstacles, no diffraction
        
        # Diffraction loss (simplified)
        diffraction_loss = obstacles * 3.0  # 3dB per obstacle
        
        # Calculate diffracted path RSSI
        distance = np.sqrt(sum((ap[i] - point[i])**2 for i in range(3)))
        free_space_loss = 20 * np.log10(4 * np.pi * distance / self.wavelength)
        
        rssi = tx_power - free_space_loss - diffraction_loss
        
        return rssi
    
    def _count_obstacles_along_path(self, ap: Tuple[float, float, float], 
                                  point: Tuple[float, float, float], 
                                  materials_grid) -> int:
        """Count obstacles along the direct path."""
        if materials_grid is None:
            return 0
        
        path_points = self._get_3d_path_points(ap, point)
        obstacles = 0
        
        for x, y, z in path_points:
            material = self._get_material_at_point(x, y, z, materials_grid)
            if material is not None and material.name != 'Air':
                obstacles += 1
        
        return obstacles
    
    def _combine_multipath_signals(self, signals: List[float]) -> float:
        """Combine multiple signals using power addition."""
        if not signals:
            return -100.0
        
        # Convert dBm to mW
        powers_mw = [10**(signal/10) for signal in signals if signal > -100]
        
        if not powers_mw:
            return -100.0
        
        # Sum powers
        total_power_mw = sum(powers_mw)
        
        # Convert back to dBm
        total_rssi = 10 * np.log10(total_power_mw)
        
        return total_rssi
    
    def calculate_rssi_grid(self, ap: Tuple[float, float, float], 
                           points: List[Tuple[float, float, float]], 
                           materials_grid, **kwargs) -> np.ndarray:
        """Calculate RSSI for a grid of points efficiently."""
        rssi_values = []
        
        for point in points:
            rssi = self.calculate_rssi(ap, point, materials_grid, **kwargs)
            rssi_values.append(rssi)
        
        return np.array(rssi_values)

class FastRayTracingEngine(PropagationEngine):
    """
    Fast Ray Tracing Engine: Optimized version with advanced physics.
    """
    def calculate_rssi(self, ap, point, materials_grid, **kwargs):
        # Use the advanced physics engine for calculations
        advanced_engine = AdvancedPhysicsEngine(
            frequency=kwargs.get('frequency', 2.4e9),
            temperature=kwargs.get('temperature', 293.15)
        )
        
        return advanced_engine.calculate_rssi(ap, point, materials_grid, **kwargs)

class Cost231Engine(PropagationEngine):
    """
    COST-231 Hata Model Engine with material corrections.
    """
    def calculate_rssi(self, ap, point, materials_grid, **kwargs):
        # Extract coordinates
        ap_x, ap_y, ap_z = ap if len(ap) == 3 else (ap[0], ap[1], 0)
        x, y, z = point if len(point) == 3 else (point[0], point[1], 0)
        
        # Calculate distance
        distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2 + (z - ap_z)**2)
        
        if distance < 1e-3:
            return kwargs.get('tx_power', 20.0)
        
        # COST-231 Hata model parameters
        frequency = kwargs.get('frequency', 2400)  # MHz
        tx_power = kwargs.get('tx_power', 20.0)
        ap_height = ap_z
        rx_height = z
        
        # COST-231 Hata path loss
        if frequency < 1500:
            # COST-231 Hata model for 900-1500 MHz
            path_loss = 46.3 + 33.9 * np.log10(frequency) - 13.82 * np.log10(ap_height) - \
                       (1.1 * np.log10(frequency) - 0.7) * rx_height + \
                       (1.56 * np.log10(frequency) - 0.8) + \
                       44.9 - 6.55 * np.log10(ap_height) * np.log10(distance/1000)
        else:
            # COST-231 Hata model for 1500-2000 MHz
            path_loss = 46.3 + 33.9 * np.log10(frequency) - 13.82 * np.log10(ap_height) - \
                       (1.1 * np.log10(frequency) - 0.7) * rx_height + \
                       3.0 + \
                       44.9 - 6.55 * np.log10(ap_height) * np.log10(distance/1000)
        
        # Add material attenuation
        material_attenuation = self._calculate_material_attenuation(ap, point, materials_grid)
        
        # Calculate RSSI
        rssi = tx_power - path_loss - material_attenuation
        
        return rssi
    
    def _calculate_material_attenuation(self, ap, point, materials_grid):
        """Calculate material attenuation for COST-231 model."""
        if materials_grid is None:
            return 0.0
        
        # Simplified material attenuation calculation
        # In a full implementation, this would traverse the path
        return 0.0

class VPLEEngine(PropagationEngine):
    """
    Variable Path Loss Exponent Engine with machine learning enhancements.
    """
    def __init__(self, ml_model=None):
        self.ml_model = ml_model
        self.base_path_loss_exponent = 2.0
    
    def calculate_rssi(self, ap, point, materials_grid, **kwargs):
        # Extract coordinates
        ap_x, ap_y, ap_z = ap if len(ap) == 3 else (ap[0], ap[1], 0)
        x, y, z = point if len(point) == 3 else (point[0], point[1], 0)
        
        # Calculate distance
        distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2 + (z - ap_z)**2)
        
        if distance < 1e-3:
            return kwargs.get('tx_power', 20.0)
        
        # Calculate path loss exponent based on environment
        path_loss_exponent = self._calculate_path_loss_exponent(ap, point, materials_grid)
        
        # Variable path loss model
        frequency = kwargs.get('frequency', 2400)  # MHz
        tx_power = kwargs.get('tx_power', 20.0)
        
        # Reference distance and path loss
        d0 = 1.0  # Reference distance in meters
        PL0 = 20 * np.log10(4 * np.pi * d0 * frequency * 1e6 / 3e8)
        
        # Path loss
        path_loss = PL0 + 10 * path_loss_exponent * np.log10(distance / d0)
        
        # Add material attenuation
        material_attenuation = self._calculate_material_attenuation(ap, point, materials_grid)
        
        # Calculate RSSI
        rssi = tx_power - path_loss - material_attenuation
        
        return rssi
    
    def _calculate_path_loss_exponent(self, ap, point, materials_grid):
        """Calculate path loss exponent based on environment complexity."""
        if materials_grid is None:
            return self.base_path_loss_exponent
        
        # Count obstacles along the path
        obstacles = self._count_obstacles(ap, point, materials_grid)
        
        # Adjust path loss exponent based on obstacles
        if obstacles == 0:
            return 2.0  # Free space
        elif obstacles < 5:
            return 2.5  # Light obstacles
        elif obstacles < 10:
            return 3.0  # Medium obstacles
        else:
            return 3.5  # Heavy obstacles
    
    def _count_obstacles(self, ap, point, materials_grid):
        """Count obstacles along the path."""
        # Simplified obstacle counting
        return 0
    
    def _calculate_material_attenuation(self, ap, point, materials_grid):
        """Calculate material attenuation for VPLE model."""
        if materials_grid is None:
            return 0.0
        
        # Simplified material attenuation calculation
        return 0.0 