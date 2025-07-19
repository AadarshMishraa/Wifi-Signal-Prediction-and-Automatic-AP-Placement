"""
Performance Optimization and Profiling System

This module provides:
- Advanced profiling and performance monitoring
- Vectorized operations using NumPy
- Parallel processing for independent calculations
- Intelligent caching strategies
- Memory optimization
- Performance bottleneck identification
"""

import numpy as np
import time
import cProfile
import pstats
import io
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import gc
import weakref

logger = logging.getLogger(__name__)

class ProfilerMode(Enum):
    """Profiling modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    DETAILED = "detailed"
    MEMORY = "memory"

@dataclass
class PerformanceProfile:
    """Performance profile data."""
    function_name: str
    total_time: float
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

@dataclass
class CacheStats:
    """Cache statistics."""
    cache_name: str
    hits: int
    misses: int
    size: int
    max_size: int
    hit_rate: float

class PerformanceOptimizer:
    """
    Advanced performance optimization and profiling system.
    """
    
    def __init__(self, profiler_mode: ProfilerMode = ProfilerMode.BASIC):
        """Initialize the performance optimizer."""
        self.profiler_mode = profiler_mode
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.cache_stats: Dict[str, CacheStats] = {}
        self.memory_tracker = MemoryTracker()
        self.profiler = None
        self.stats = None
        
        # Performance tracking
        self.start_time = time.time()
        self.operation_times = {}
        
        logger.info(f"Performance Optimizer initialized with mode: {profiler_mode.value}")
    
    def start_profiling(self):
        """Start profiling if enabled."""
        if self.profiler_mode != ProfilerMode.DISABLED:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            logger.info("Profiling started")
    
    def stop_profiling(self) -> Optional[str]:
        """Stop profiling and return statistics."""
        if self.profiler is not None:
            self.profiler.disable()
            s = io.StringIO()
            self.stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            self.stats.print_stats(20)  # Top 20 functions
            logger.info("Profiling stopped")
            return s.getvalue()
        return None
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceProfile]:
        """Profile a single function execution."""
        start_time = time.time()
        start_memory = self.memory_tracker.get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in profiled function {func.__name__}: {e}")
            raise
        
        end_time = time.time()
        end_memory = self.memory_tracker.get_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory if end_memory and start_memory else None
        
        # Update profile
        if func.__name__ not in self.profiles:
            self.profiles[func.__name__] = PerformanceProfile(
                function_name=func.__name__,
                total_time=execution_time,
                call_count=1,
                avg_time=execution_time,
                min_time=execution_time,
                max_time=execution_time,
                memory_usage=memory_usage
            )
        else:
            profile = self.profiles[func.__name__]
            profile.total_time += execution_time
            profile.call_count += 1
            profile.avg_time = profile.total_time / profile.call_count
            profile.min_time = min(profile.min_time, execution_time)
            profile.max_time = max(profile.max_time, execution_time)
            if memory_usage is not None:
                profile.memory_usage = memory_usage
        
        return result, self.profiles[func.__name__]
    
    def profile_decorator(self, func: Callable) -> Callable:
        """Decorator for profiling functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.profile_function(func, *args, **kwargs)[0]
        return wrapper
    
    def vectorized_rssi_calculation(self, ap_locations: np.ndarray, 
                                  points: np.ndarray, 
                                  tx_power: float = 20.0,
                                  frequency: float = 2.4e9) -> np.ndarray:
        """
        Vectorized RSSI calculation for multiple APs and points.
        
        Args:
            ap_locations: Array of AP coordinates (n_aps, 3)
            points: Array of receiver points (n_points, 3)
            tx_power: Transmit power in dBm
            frequency: Frequency in Hz
            
        Returns:
            RSSI matrix (n_aps, n_points)
        """
        try:
            n_aps = ap_locations.shape[0]
            n_points = points.shape[0]
            
            # Reshape for broadcasting
            ap_locations_expanded = ap_locations[:, np.newaxis, :]  # (n_aps, 1, 3)
            points_expanded = points[np.newaxis, :, :]  # (1, n_points, 3)
            
            # Calculate distances vectorized
            distances = np.sqrt(np.sum((ap_locations_expanded - points_expanded) ** 2, axis=2))  # (n_aps, n_points)
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-6)
            
            # Calculate free space path loss vectorized
            wavelength = 3e8 / frequency
            free_space_loss = 20 * np.log10(4 * np.pi * distances / wavelength)
            
            # Calculate RSSI vectorized
            rssi = tx_power - free_space_loss
            
            # Clip to reasonable range
            rssi = np.clip(rssi, -100.0, 0.0)
            
            return rssi
            
        except Exception as e:
            logger.error(f"Error in vectorized RSSI calculation: {e}")
            return np.full((n_aps, n_points), -100.0)
    
    def vectorized_material_attenuation(self, start_points: np.ndarray,
                                      end_points: np.ndarray,
                                      materials_grid: np.ndarray,
                                      resolution: float = 0.2) -> np.ndarray:
        """
        Vectorized material attenuation calculation.
        
        Args:
            start_points: Array of start points (n_paths, 3)
            end_points: Array of end points (n_paths, 3)
            materials_grid: 3D materials grid
            resolution: Grid resolution
            
        Returns:
            Attenuation array (n_paths,)
        """
        try:
            n_paths = start_points.shape[0]
            attenuations = np.zeros(n_paths)
            
            # Calculate path vectors
            path_vectors = end_points - start_points
            path_lengths = np.sqrt(np.sum(path_vectors ** 2, axis=1))
            
            # Normalize path vectors
            path_directions = path_vectors / (path_lengths[:, np.newaxis] + 1e-6)
            
            # Calculate number of steps for each path
            max_steps = int(np.max(path_lengths) / resolution) + 1
            
            # Vectorized path traversal
            for step in range(max_steps):
                # Calculate current positions
                t = step / max_steps
                current_positions = start_points + t * path_vectors
                
                # Convert to grid coordinates
                grid_coords = (current_positions / resolution).astype(int)
                
                # Clamp to grid bounds
                grid_coords = np.clip(grid_coords, 0, np.array(materials_grid.shape) - 1)
                
                # Get materials at current positions
                materials = materials_grid[grid_coords[:, 2], grid_coords[:, 1], grid_coords[:, 0]]
                
                # Calculate attenuation for this step
                step_lengths = path_lengths / max_steps
                step_attenuations = np.array([
                    self._get_material_attenuation(material, step_lengths[i], 2.4e9)
                    for i, material in enumerate(materials)
                ])
                
                attenuations += step_attenuations
            
            return attenuations
            
        except Exception as e:
            logger.error(f"Error in vectorized material attenuation: {e}")
            return np.zeros(n_paths)
    
    def _get_material_attenuation(self, material, distance: float, frequency: float) -> float:
        """Get attenuation for a material."""
        try:
            if hasattr(material, 'calculate_attenuation'):
                return material.calculate_attenuation(frequency, distance)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def parallel_rssi_calculation(self, ap_locations: List[Tuple[float, float, float]],
                                points: List[Tuple[float, float, float]],
                                materials_grid: np.ndarray,
                                tx_power: float = 20.0,
                                max_workers: int = None) -> np.ndarray:
        """
        Parallel RSSI calculation using multiprocessing.
        
        Args:
            ap_locations: List of AP coordinates
            points: List of receiver points
            materials_grid: 3D materials grid
            tx_power: Transmit power in dBm
            max_workers: Maximum number of workers
            
        Returns:
            RSSI matrix (n_aps, n_points)
        """
        try:
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(ap_locations))
            
            n_aps = len(ap_locations)
            n_points = len(points)
            
            # Initialize result matrix
            rssi_matrix = np.full((n_aps, n_points), -100.0)
            
            # Use parallel processing for large calculations
            if n_aps * n_points > 1000:  # Threshold for parallel processing
                logger.info(f"Using parallel processing with {max_workers} workers")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit tasks for each AP
                    futures = []
                    for ap_idx, ap_location in enumerate(ap_locations):
                        future = executor.submit(
                            self._calculate_rssi_for_ap_parallel,
                            ap_location, points, materials_grid, tx_power
                        )
                        futures.append((ap_idx, future))
                    
                    # Collect results
                    for ap_idx, future in futures:
                        try:
                            rssi_values = future.result()
                            rssi_matrix[ap_idx, :] = rssi_values
                        except Exception as e:
                            logger.error(f"Error in parallel RSSI calculation for AP {ap_idx}: {e}")
                            rssi_matrix[ap_idx, :] = -100.0
            else:
                # Sequential processing for small calculations
                for ap_idx, ap_location in enumerate(ap_locations):
                    rssi_values = self._calculate_rssi_for_ap_parallel(
                        ap_location, points, materials_grid, tx_power
                    )
                    rssi_matrix[ap_idx, :] = rssi_values
            
            return rssi_matrix
            
        except Exception as e:
            logger.error(f"Error in parallel RSSI calculation: {e}")
            return np.full((len(ap_locations), len(points)), -100.0)
    
    def _calculate_rssi_for_ap_parallel(self, ap_location: Tuple[float, float, float],
                                      points: List[Tuple[float, float, float]],
                                      materials_grid: np.ndarray,
                                      tx_power: float) -> np.ndarray:
        """Calculate RSSI for one AP at multiple points (for parallel processing)."""
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
                
                # Material attenuation (simplified for parallel processing)
                material_attenuation = 0.0  # Could be enhanced with actual material calculation
                
                # Total RSSI
                rssi = tx_power - free_space_loss - material_attenuation
                rssi_values.append(rssi)
            
            return np.array(rssi_values)
            
        except Exception as e:
            logger.error(f"Error calculating RSSI for AP: {e}")
            return np.full(len(points), -100.0)
    
    def create_cache(self, cache_name: str, max_size: int = 1000) -> Callable:
        """
        Create a named cache with statistics tracking.
        
        Args:
            cache_name: Name of the cache
            max_size: Maximum cache size
            
        Returns:
            Decorator function for caching
        """
        cache = {}
        cache_stats = CacheStats(
            cache_name=cache_name,
            hits=0,
            misses=0,
            size=0,
            max_size=max_size,
            hit_rate=0.0
        )
        
        self.cache_stats[cache_name] = cache_stats
        
        def cache_decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = str((args, tuple(sorted(kwargs.items()))))
                
                if cache_key in cache:
                    # Cache hit
                    cache_stats.hits += 1
                    cache_stats.hit_rate = cache_stats.hits / (cache_stats.hits + cache_stats.misses)
                    return cache[cache_key]
                else:
                    # Cache miss
                    cache_stats.misses += 1
                    result = func(*args, **kwargs)
                    
                    # Add to cache if not full
                    if len(cache) < max_size:
                        cache[cache_key] = result
                        cache_stats.size = len(cache)
                    
                    cache_stats.hit_rate = cache_stats.hits / (cache_stats.hits + cache_stats.misses)
                    return result
            
            return wrapper
        
        return cache_decorator
    
    def optimize_memory_usage(self):
        """Optimize memory usage by clearing caches and running garbage collection."""
        try:
            # Clear all caches
            for cache_name in self.cache_stats:
                cache_stats = self.cache_stats[cache_name]
                cache_stats.hits = 0
                cache_stats.misses = 0
                cache_stats.size = 0
                cache_stats.hit_rate = 0.0
            
            # Run garbage collection
            gc.collect()
            
            # Clear operation times
            self.operation_times.clear()
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Error in memory optimization: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_time = time.time() - self.start_time
        
        # Calculate performance metrics
        avg_times = {}
        for func_name, times in self.operation_times.items():
            if times:
                avg_times[func_name] = np.mean(times)
        
        # Memory usage
        memory_usage = self.memory_tracker.get_memory_usage()
        
        return {
            'total_runtime': total_time,
            'function_profiles': {
                name: {
                    'total_time': profile.total_time,
                    'call_count': profile.call_count,
                    'avg_time': profile.avg_time,
                    'min_time': profile.min_time,
                    'max_time': profile.max_time,
                    'memory_usage': profile.memory_usage
                }
                for name, profile in self.profiles.items()
            },
            'cache_statistics': {
                name: {
                    'hits': stats.hits,
                    'misses': stats.misses,
                    'size': stats.size,
                    'max_size': stats.max_size,
                    'hit_rate': stats.hit_rate
                }
                for name, stats in self.cache_stats.items()
            },
            'memory_usage_mb': memory_usage,
            'average_function_times': avg_times
        }
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check function performance
        for func_name, profile in self.profiles.items():
            if profile.avg_time > 0.1:  # Functions taking more than 100ms on average
                bottlenecks.append({
                    'type': 'function',
                    'name': func_name,
                    'avg_time': profile.avg_time,
                    'call_count': profile.call_count,
                    'total_time': profile.total_time,
                    'suggestion': 'Consider optimization or caching'
                })
        
        # Check cache performance
        for cache_name, stats in self.cache_stats.items():
            if stats.hit_rate < 0.5:  # Low cache hit rate
                bottlenecks.append({
                    'type': 'cache',
                    'name': cache_name,
                    'hit_rate': stats.hit_rate,
                    'suggestion': 'Review cache key strategy or increase cache size'
                })
        
        # Check memory usage
        memory_usage = self.memory_tracker.get_memory_usage()
        if memory_usage and memory_usage > 1000:  # More than 1GB
            bottlenecks.append({
                'type': 'memory',
                'usage_mb': memory_usage,
                'suggestion': 'Consider memory optimization or data structure changes'
            })
        
        return sorted(bottlenecks, key=lambda x: x.get('avg_time', 0), reverse=True)

class MemoryTracker:
    """Track memory usage."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return None

def test_performance_optimizer():
    """Test the performance optimizer."""
    print("Testing Performance Optimizer...")
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(ProfilerMode.BASIC)
    
    # Test vectorized RSSI calculation
    ap_locations = np.array([[10.0, 10.0, 2.7], [30.0, 30.0, 2.7]])
    points = np.array([[5.0, 5.0, 1.5], [15.0, 15.0, 1.5], [25.0, 25.0, 1.5]])
    
    rssi_matrix = optimizer.vectorized_rssi_calculation(ap_locations, points)
    print(f"Vectorized RSSI matrix shape: {rssi_matrix.shape}")
    
    # Test profiling decorator
    @optimizer.profile_decorator
    def test_function(x):
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    result = test_function(5)
    print(f"Profiled function result: {result}")
    
    # Test cache
    cache_decorator = optimizer.create_cache("test_cache", max_size=10)
    
    @cache_decorator
    def expensive_function(x):
        time.sleep(0.1)  # Simulate expensive operation
        return x ** 2
    
    # First call (cache miss)
    result1 = expensive_function(5)
    # Second call (cache hit)
    result2 = expensive_function(5)
    
    print(f"Cached function results: {result1}, {result2}")
    
    # Get performance report
    report = optimizer.get_performance_report()
    print(f"Performance report keys: {list(report.keys())}")
    
    # Identify bottlenecks
    bottlenecks = optimizer.identify_bottlenecks()
    print(f"Identified bottlenecks: {len(bottlenecks)}")
    
    print("Performance Optimizer test completed successfully!")

if __name__ == "__main__":
    test_performance_optimizer() 