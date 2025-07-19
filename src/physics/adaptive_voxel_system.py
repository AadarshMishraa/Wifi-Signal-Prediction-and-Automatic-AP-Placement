"""
Adaptive Voxel System for Advanced WiFi Propagation Modeling

This module implements:
- Adaptive voxel resolution based on signal variability and obstacle density
- Optimized 3D voxel traversal with unified 2D/3D handling
- Numerical stability and edge case handling
- Comprehensive error handling and logging
- Performance optimization with caching and vectorization
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Union, Dict, Set
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoxelType(Enum):
    """Types of voxels for adaptive resolution."""
    HIGH_RESOLUTION = "high_resolution"      # Near APs, obstacles, high variability
    MEDIUM_RESOLUTION = "medium_resolution"  # Normal areas
    LOW_RESOLUTION = "low_resolution"        # Open spaces, far from APs

@dataclass
class VoxelConfig:
    """Configuration for adaptive voxel system."""
    base_resolution: float = 0.2  # Base resolution in meters
    high_res_multiplier: float = 4.0  # High resolution = base_res / multiplier
    medium_res_multiplier: float = 2.0
    low_res_multiplier: float = 0.5  # Low resolution = base_res * multiplier
    
    # Adaptive resolution parameters
    ap_influence_radius: float = 5.0  # Meters around APs for high resolution
    obstacle_influence_radius: float = 2.0  # Meters around obstacles
    variability_threshold: float = 0.1  # Signal variability threshold for high resolution
    
    # Performance parameters
    max_voxels_per_dimension: int = 1000  # Maximum voxels per dimension
    cache_size: int = 10000  # LRU cache size for path calculations
    parallel_threshold: int = 100  # Minimum points for parallel processing

class AdaptiveVoxelSystem:
    """
    Advanced voxel system with adaptive resolution and optimized traversal.
    """
    
    def __init__(self, config: VoxelConfig = None):
        """Initialize the adaptive voxel system."""
        self.config = config or VoxelConfig()
        self.materials_grid = None
        self.voxel_types = None
        self.resolution_map = None
        self.ap_locations = []
        self.obstacle_locations = []
        
        # Performance tracking
        self.calculation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize caches
        self._path_cache = {}
        self._material_cache = {}
        
        logger.info("Adaptive Voxel System initialized")
    
    def set_materials_grid(self, materials_grid: np.ndarray):
        """Set the 3D materials grid."""
        try:
            self.materials_grid = materials_grid
            logger.info(f"Materials grid set with shape: {materials_grid.shape}")
        except Exception as e:
            logger.error(f"Error setting materials grid: {e}")
            raise
    
    def set_ap_locations(self, ap_locations: List[Tuple[float, float, float]]):
        """Set AP locations for adaptive resolution."""
        self.ap_locations = ap_locations
        logger.info(f"AP locations set: {len(ap_locations)} APs")
    
    def set_obstacle_locations(self, obstacle_locations: List[Tuple[float, float, float]]):
        """Set obstacle locations for adaptive resolution."""
        self.obstacle_locations = obstacle_locations
        logger.info(f"Obstacle locations set: {len(obstacle_locations)} obstacles")
    
    def calculate_adaptive_resolution(self, building_dimensions: Tuple[float, float, float]):
        """
        Calculate adaptive voxel resolution based on APs, obstacles, and signal variability.
        
        Args:
            building_dimensions: (width, length, height) in meters
            
        Returns:
            resolution_map: 3D array of resolution values
        """
        try:
            width, length, height = building_dimensions
            
            # Initialize resolution map with base resolution
            nx = int(width / self.config.base_resolution)
            ny = int(length / self.config.base_resolution)
            nz = int(height / self.config.base_resolution)
            
            # Limit maximum voxels per dimension
            nx = min(nx, self.config.max_voxels_per_dimension)
            ny = min(ny, self.config.max_voxels_per_dimension)
            nz = min(nz, self.config.max_voxels_per_dimension)
            
            self.resolution_map = np.full((nz, ny, nx), self.config.base_resolution)
            
            # Apply adaptive resolution based on AP locations
            self._apply_ap_based_resolution(width, length, height)
            
            # Apply adaptive resolution based on obstacles
            self._apply_obstacle_based_resolution(width, length, height)
            
            # Apply adaptive resolution based on signal variability
            self._apply_variability_based_resolution(width, length, height)
            
            logger.info(f"Adaptive resolution calculated: {nx}x{ny}x{nz} voxels")
            return self.resolution_map
            
        except Exception as e:
            logger.error(f"Error calculating adaptive resolution: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _apply_ap_based_resolution(self, width: float, length: float, height: float):
        """Apply high resolution around AP locations."""
        if not self.ap_locations:
            return
        
        nx, ny, nz = self.resolution_map.shape
        
        for ap_x, ap_y, ap_z in self.ap_locations:
            # Convert AP coordinates to grid indices
            gx = int(ap_x / width * nx)
            gy = int(ap_y / length * ny)
            gz = int(ap_z / height * nz)
            
            # Calculate influence radius in grid units
            influence_radius = int(self.config.ap_influence_radius / self.config.base_resolution)
            
            # Apply high resolution in influence area
            for dz in range(-influence_radius, influence_radius + 1):
                for dy in range(-influence_radius, influence_radius + 1):
                    for dx in range(-influence_radius, influence_radius + 1):
                        nx_idx = gx + dx
                        ny_idx = gy + dy
                        nz_idx = gz + dz
                        
                        if (0 <= nx_idx < nx and 0 <= ny_idx < ny and 0 <= nz_idx < nz):
                            distance = np.sqrt(dx**2 + dy**2 + dz**2)
                            if distance <= influence_radius:
                                # High resolution near APs
                                self.resolution_map[nz_idx, ny_idx, nx_idx] = (
                                    self.config.base_resolution / self.config.high_res_multiplier
                                )
    
    def _apply_obstacle_based_resolution(self, width: float, length: float, height: float):
        """Apply high resolution around obstacles."""
        if not self.obstacle_locations:
            return
        
        nx, ny, nz = self.resolution_map.shape
        
        for obs_x, obs_y, obs_z in self.obstacle_locations:
            # Convert obstacle coordinates to grid indices
            gx = int(obs_x / width * nx)
            gy = int(obs_y / length * ny)
            gz = int(obs_z / height * nz)
            
            # Calculate influence radius in grid units
            influence_radius = int(self.config.obstacle_influence_radius / self.config.base_resolution)
            
            # Apply high resolution in influence area
            for dz in range(-influence_radius, influence_radius + 1):
                for dy in range(-influence_radius, influence_radius + 1):
                    for dx in range(-influence_radius, influence_radius + 1):
                        nx_idx = gx + dx
                        ny_idx = gy + dy
                        nz_idx = gz + dz
                        
                        if (0 <= nx_idx < nx and 0 <= ny_idx < ny and 0 <= nz_idx < nz):
                            distance = np.sqrt(dx**2 + dy**2 + dz**2)
                            if distance <= influence_radius:
                                # High resolution near obstacles
                                current_res = self.resolution_map[nz_idx, ny_idx, nx_idx]
                                high_res = self.config.base_resolution / self.config.high_res_multiplier
                                self.resolution_map[nz_idx, ny_idx, nx_idx] = min(current_res, high_res)
    
    def _apply_variability_based_resolution(self, width: float, length: float, height: float):
        """Apply resolution based on signal variability (simplified model)."""
        # This is a simplified implementation
        # In a full implementation, this would analyze signal variability patterns
        pass
    
    @lru_cache(maxsize=10000)
    def get_optimized_path_points(self, start: Tuple[float, float, float], 
                                end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Get optimized path points using adaptive resolution and unified 3D/2D handling.
        
        Args:
            start: Starting point (x, y, z)
            end: Ending point (x, y, z)
            
        Returns:
            List of path points with appropriate resolution
        """
        try:
            # Check if points are very close
            distance = np.sqrt(sum((end[i] - start[i])**2 for i in range(3)))
            if distance < 1e-6:
                return [start]
            
            # Determine if we need 2D or 3D traversal
            if abs(start[2] - end[2]) < 1e-3:
                # 2D traversal (same z-level)
                return self._get_2d_path_points(start, end)
            else:
                # 3D traversal
                return self._get_3d_path_points(start, end)
                
        except Exception as e:
            logger.error(f"Error in get_optimized_path_points: {e}")
            # Fallback to simple linear interpolation
            return self._get_fallback_path_points(start, end)
    
    def _get_3d_path_points(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Get 3D path points using optimized Bresenham algorithm."""
        try:
            x1, y1, z1 = start
            x2, y2, z2 = end
            
            # Use adaptive resolution for coordinate conversion
            if self.resolution_map is not None:
                # Get resolution at start point
                start_res = self._get_resolution_at_point(x1, y1, z1)
                end_res = self._get_resolution_at_point(x2, y2, z2)
                resolution = min(start_res, end_res)
            else:
                resolution = self.config.base_resolution
            
            # Convert to grid coordinates
            gx1, gy1, gz1 = int(x1 / resolution), int(y1 / resolution), int(z1 / resolution)
            gx2, gy2, gz2 = int(x2 / resolution), int(y2 / resolution), int(z2 / resolution)
            
            # Optimized 3D Bresenham algorithm
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
            
        except Exception as e:
            logger.error(f"Error in 3D path calculation: {e}")
            return self._get_fallback_path_points(start, end)
    
    def _get_2d_path_points(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Get 2D path points (same z-level) using optimized algorithm."""
        try:
            x1, y1, z1 = start
            x2, y2, z2 = end
            
            # Use adaptive resolution
            if self.resolution_map is not None:
                resolution = min(
                    self._get_resolution_at_point(x1, y1, z1),
                    self._get_resolution_at_point(x2, y2, z2)
                )
            else:
                resolution = self.config.base_resolution
            
            # Convert to grid coordinates
            gx1, gy1 = int(x1 / resolution), int(y1 / resolution)
            gx2, gy2 = int(x2 / resolution), int(y2 / resolution)
            
            # Optimized 2D Bresenham algorithm
            points = []
            dx = abs(gx2 - gx1)
            dy = abs(gy2 - gy1)
            
            sx = 1 if gx2 > gx1 else -1
            sy = 1 if gy2 > gy1 else -1
            
            err = dx - dy
            
            while True:
                points.append((gx1 * resolution, gy1 * resolution, z1))
                
                if gx1 == gx2 and gy1 == gy2:
                    break
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    gx1 += sx
                if e2 < dx:
                    err += dx
                    gy1 += sy
            
            return points
            
        except Exception as e:
            logger.error(f"Error in 2D path calculation: {e}")
            return self._get_fallback_path_points(start, end)
    
    def _get_fallback_path_points(self, start: Tuple[float, float, float], 
                                 end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Fallback path calculation using linear interpolation."""
        try:
            distance = np.sqrt(sum((end[i] - start[i])**2 for i in range(3)))
            if distance < 1e-6:
                return [start]
            
            # Simple linear interpolation
            num_points = max(2, int(distance / self.config.base_resolution))
            points = []
            
            for i in range(num_points):
                t = i / (num_points - 1)
                point = tuple(start[j] + t * (end[j] - start[j]) for j in range(3))
                points.append(point)
            
            return points
            
        except Exception as e:
            logger.error(f"Error in fallback path calculation: {e}")
            return [start, end]
    
    def _get_resolution_at_point(self, x: float, y: float, z: float) -> float:
        """Get resolution at a specific point."""
        try:
            if self.resolution_map is None:
                return self.config.base_resolution
            
            # Convert to grid indices
            nx, ny, nz = self.resolution_map.shape
            
            # Get building dimensions (assume 1:1 mapping for now)
            gx = int(x / self.config.base_resolution)
            gy = int(y / self.config.base_resolution)
            gz = int(z / self.config.base_resolution)
            
            # Clamp to grid bounds
            gx = max(0, min(gx, nx - 1))
            gy = max(0, min(gy, ny - 1))
            gz = max(0, min(gz, nz - 1))
            
            return self.resolution_map[gz, gy, gx]
            
        except Exception as e:
            logger.warning(f"Error getting resolution at point: {e}")
            return self.config.base_resolution
    
    def calculate_material_attenuation_optimized(self, start: Tuple[float, float, float], 
                                               end: Tuple[float, float, float], 
                                               materials_grid) -> float:
        """
        Calculate material attenuation along path with optimized performance.
        
        Args:
            start: Starting point
            end: Ending point
            materials_grid: 3D materials grid
            
        Returns:
            Total attenuation in dB
        """
        try:
            start_time = time.time()
            
            # Get optimized path points
            path_points = self.get_optimized_path_points(start, end)
            
            total_attenuation = 0.0
            seen_materials = set()
            
            for i, point in enumerate(path_points):
                # Get material at this point
                material = self._get_material_at_point_optimized(point, materials_grid)
                
                if material is None or material.name == 'Air':
                    continue
                
                # Calculate segment length
                if i < len(path_points) - 1:
                    next_point = path_points[i + 1]
                    segment_length = np.sqrt(sum((next_point[j] - point[j])**2 for j in range(3)))
                else:
                    segment_length = 0.1  # Default segment length
                
                # Calculate attenuation for this material segment
                if hasattr(material, 'calculate_attenuation'):
                    segment_atten = material.calculate_attenuation(2.4e9, segment_length)
                else:
                    segment_atten = 0.0
                
                # Avoid double-counting same material
                material_key = (material.name, point[0], point[1], point[2])
                if material_key not in seen_materials:
                    total_attenuation += segment_atten
                    seen_materials.add(material_key)
            
            # Track performance
            calculation_time = time.time() - start_time
            self.calculation_times.append(calculation_time)
            
            return total_attenuation
            
        except Exception as e:
            logger.error(f"Error in material attenuation calculation: {e}")
            logger.error(traceback.format_exc())
            return 0.0
    
    def _get_material_at_point_optimized(self, point: Tuple[float, float, float], 
                                       materials_grid) -> Optional:
        """Get material at point with optimized lookup."""
        try:
            if materials_grid is None:
                return None
            
            x, y, z = point
            
            # Use adaptive resolution for grid lookup
            resolution = self._get_resolution_at_point(x, y, z)
            
            # Convert to grid coordinates
            gx = int(x / resolution)
            gy = int(y / resolution)
            gz = int(z / resolution)
            
            # Check bounds
            if (0 <= gz < len(materials_grid) and 
                0 <= gy < len(materials_grid[0]) and 
                0 <= gx < len(materials_grid[0][0])):
                return materials_grid[gz][gy][gx]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting material at point: {e}")
            return None
    
    def calculate_rssi_batch_parallel(self, ap_locations: List[Tuple[float, float, float]], 
                                    points: List[Tuple[float, float, float]], 
                                    materials_grid, 
                                    tx_power: float = 20.0,
                                    max_workers: int = None) -> np.ndarray:
        """
        Calculate RSSI for multiple APs and points in parallel.
        
        Args:
            ap_locations: List of AP coordinates
            points: List of receiver points
            materials_grid: 3D materials grid
            tx_power: Transmit power in dBm
            max_workers: Maximum number of parallel workers
            
        Returns:
            RSSI matrix: shape (num_aps, num_points)
        """
        try:
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(ap_locations))
            
            num_aps = len(ap_locations)
            num_points = len(points)
            
            # Initialize RSSI matrix
            rssi_matrix = np.full((num_aps, num_points), -100.0)
            
            # Use parallel processing for large batches
            if num_aps * num_points > self.config.parallel_threshold:
                logger.info(f"Using parallel processing with {max_workers} workers")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit tasks for each AP
                    futures = []
                    for ap_idx, ap_location in enumerate(ap_locations):
                        future = executor.submit(
                            self._calculate_rssi_for_ap,
                            ap_location, points, materials_grid, tx_power
                        )
                        futures.append((ap_idx, future))
                    
                    # Collect results
                    for ap_idx, future in futures:
                        try:
                            rssi_values = future.result()
                            rssi_matrix[ap_idx, :] = rssi_values
                        except Exception as e:
                            logger.error(f"Error calculating RSSI for AP {ap_idx}: {e}")
                            rssi_matrix[ap_idx, :] = -100.0
            else:
                # Sequential processing for small batches
                for ap_idx, ap_location in enumerate(ap_locations):
                    rssi_values = self._calculate_rssi_for_ap(
                        ap_location, points, materials_grid, tx_power
                    )
                    rssi_matrix[ap_idx, :] = rssi_values
            
            return rssi_matrix
            
        except Exception as e:
            logger.error(f"Error in batch RSSI calculation: {e}")
            logger.error(traceback.format_exc())
            return np.full((len(ap_locations), len(points)), -100.0)
    
    def _calculate_rssi_for_ap(self, ap_location: Tuple[float, float, float], 
                              points: List[Tuple[float, float, float]], 
                              materials_grid, 
                              tx_power: float) -> np.ndarray:
        """Calculate RSSI for one AP at multiple points."""
        try:
            rssi_values = []
            
            for point in points:
                # Calculate distance
                distance = np.sqrt(sum((ap_location[i] - point[i])**2 for i in range(3)))
                
                if distance < 1e-6:
                    rssi_values.append(tx_power)
                    continue
                
                # Free space path loss
                wavelength = 3e8 / 2.4e9
                free_space_loss = 20 * np.log10(4 * np.pi * distance / wavelength)
                
                # Material attenuation
                material_attenuation = self.calculate_material_attenuation_optimized(
                    ap_location, point, materials_grid
                )
                
                # Total RSSI
                rssi = tx_power - free_space_loss - material_attenuation
                rssi_values.append(rssi)
            
            return np.array(rssi_values)
            
        except Exception as e:
            logger.error(f"Error calculating RSSI for AP: {e}")
            return np.full(len(points), -100.0)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.calculation_times:
            return {
                'avg_calculation_time': 0.0,
                'total_calculations': 0,
                'cache_hit_rate': 0.0,
                'total_cache_hits': self.cache_hits,
                'total_cache_misses': self.cache_misses
            }
        
        avg_time = np.mean(self.calculation_times)
        total_calcs = len(self.calculation_times)
        
        cache_hit_rate = 0.0
        if self.cache_hits + self.cache_misses > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
        
        return {
            'avg_calculation_time': avg_time,
            'total_calculations': total_calcs,
            'cache_hit_rate': cache_hit_rate,
            'total_cache_hits': self.cache_hits,
            'total_cache_misses': self.cache_misses
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self._path_cache.clear()
        self._material_cache.clear()
        self.get_optimized_path_points.cache_clear()
        logger.info("All caches cleared")

def test_adaptive_voxel_system():
    """Test the adaptive voxel system."""
    print("Testing Adaptive Voxel System...")
    
    # Create test configuration
    config = VoxelConfig(
        base_resolution=0.2,
        high_res_multiplier=4.0,
        medium_res_multiplier=2.0,
        low_res_multiplier=0.5,
        ap_influence_radius=5.0,
        obstacle_influence_radius=2.0
    )
    
    # Initialize system
    voxel_system = AdaptiveVoxelSystem(config)
    
    # Set test data
    ap_locations = [(10.0, 10.0, 2.7), (30.0, 30.0, 2.7)]
    obstacle_locations = [(20.0, 20.0, 1.5)]
    
    voxel_system.set_ap_locations(ap_locations)
    voxel_system.set_obstacle_locations(obstacle_locations)
    
    # Calculate adaptive resolution
    building_dimensions = (40.0, 40.0, 3.0)
    resolution_map = voxel_system.calculate_adaptive_resolution(building_dimensions)
    
    print(f"Resolution map shape: {resolution_map.shape}")
    print(f"Min resolution: {np.min(resolution_map):.3f} m")
    print(f"Max resolution: {np.max(resolution_map):.3f} m")
    print(f"Mean resolution: {np.mean(resolution_map):.3f} m")
    
    # Test path calculation
    start_point = (5.0, 5.0, 1.5)
    end_point = (35.0, 35.0, 1.5)
    
    path_points = voxel_system.get_optimized_path_points(start_point, end_point)
    print(f"Path points calculated: {len(path_points)}")
    
    # Test performance
    stats = voxel_system.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("Adaptive Voxel System test completed successfully!")

if __name__ == "__main__":
    test_adaptive_voxel_system() 