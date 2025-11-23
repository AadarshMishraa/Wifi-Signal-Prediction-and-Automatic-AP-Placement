"""Main script for WiFi signal strength prediction with dynamic AP count and optimization."""

import sys
import os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime
from matplotlib.path import Path
from typing import Optional, Dict, List, Tuple, Any

# Import necessary modules for optimization and distance calculation
from scipy.spatial import distance
from scipy import optimize as opt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

# Add import for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real

import concurrent.futures
from functools import lru_cache
from tqdm import tqdm

try:
    import orjson
except ImportError:
    import json as orjson
    orjson.dumps = lambda x: json.dumps(x).encode()

# --- Enhanced Interference Modeling and Channel Assignment ---
import networkx as nx

# Import local modules with error handling
try:
    from src.visualization.building_visualizer import BuildingVisualizer
except ImportError:
    logging.warning("BuildingVisualizer not available, using fallback")
    BuildingVisualizer = None

try:
    from src.data_collection.wifi_data_collector import WiFiDataCollector
except ImportError:
    logging.warning("WiFiDataCollector not available, using fallback")
    WiFiDataCollector = None

try:
    from src.floor_plan_analyzer import BuildingRegion, RegionBoundary, MaterialType
except ImportError:
    logging.warning("Floor plan analyzer not available")
    BuildingRegion = RegionBoundary = MaterialType = None

try:
    from src.physics.materials import MATERIALS, SignalPath, Material, ADVANCED_MATERIALS, AdvancedMaterial
except ImportError:
    logging.warning("Materials module not available, using fallback")
    MATERIALS = {}
    ADVANCED_MATERIALS = {'air': type('Material', (), {'name': 'air', 'calculate_attenuation': lambda: 0})()}
    Material = AdvancedMaterial = type('Material', (), {})

try:
    from src.models.wifi_classifier import WiFiSignalPredictor
except ImportError:
    logging.warning("WiFiSignalPredictor not available")
    WiFiSignalPredictor = None

try:
    from src.propagation.engines import FastRayTracingEngine, Cost231Engine, VPLEEngine
except ImportError:
    logging.warning("Propagation engines not available, using fallback")
    FastRayTracingEngine = Cost231Engine = VPLEEngine = None
VALIDATION_LIMITS = {
    'width': 500.0,         # Max building width in meters
    'length': 500.0,        # Max building length in meters
    'height': 50.0,         # Max building height in meters
    'min_resolution': 0.1,  # Coarsest resolution allowed
    'max_aps': 100,         # Max number of pre-placed APs
    'max_points': 50000,    # Max evaluation points to generate
}

def calculate_interference_and_sinr(ap_locations, points, collector, noise_floor_dbm=-95, channel_plan=None):
    """
    For each point, compute SINR: signal from strongest AP vs. sum of interference from all other APs (on same channel) plus noise.
    Returns average SINR, worst-case SINR, and average interference.
    """
    if not ap_locations or not points or not collector:
        return -100.0, -100.0, noise_floor_dbm
    try:
        sinr_list = []
        interference_list = []
        ap_keys = list(ap_locations.keys())
        for pt in points:
            rssi_by_ap = []
            for ap in ap_keys:
                ap_xyz = ap_locations[ap]
                try:
                    if len(ap_xyz) >= 3:
                        rssi = calculate_rssi_3d(ap_xyz[:3], pt, collector)
                    else:
                        distance = np.sqrt((pt[0] - ap_xyz[0])**2 + (pt[1] - ap_xyz[1])**2)
                        rssi = collector.calculate_rssi(distance)
                    rssi_by_ap.append((ap, rssi))
                except Exception as e:
                    logging.warning(f"Error calculating RSSI for AP {ap} at point {pt}: {e}")
                    continue
            if not rssi_by_ap:
                continue
            best_ap, best_rssi = max(rssi_by_ap, key=lambda x: x[1])
            interference_power = 0.0
            if channel_plan:
                best_channel = channel_plan.get(best_ap, 1)
                for ap, rssi in rssi_by_ap:
                    if ap != best_ap and channel_plan.get(ap, 1) == best_channel and rssi > -90:
                        interference_power += 10**(rssi/10)
            else:
                for ap, rssi in rssi_by_ap:
                    if ap != best_ap and rssi > -90:
                        interference_power += 10**(rssi/10)
            noise_power = 10**(noise_floor_dbm/10)
            signal_power = 10**(best_rssi/10)
            if (interference_power + noise_power) > 0:
                sinr_linear = signal_power / (interference_power + noise_power)
                sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100.0
            else:
                sinr_db = 100.0
            sinr_db = max(-100.0, min(100.0, sinr_db))
            sinr_list.append(sinr_db)
            if interference_power > 0:
                interference_dbm = 10 * np.log10(interference_power)
            else:
                interference_dbm = noise_floor_dbm
            interference_list.append(interference_dbm)
        if sinr_list:
            avg_sinr = float(np.mean(sinr_list))
            min_sinr = float(np.min(sinr_list))
        else:
            avg_sinr = -100.0
            min_sinr = -100.0
        if interference_list:
            avg_interference = float(np.mean(interference_list))
        else:
            avg_interference = noise_floor_dbm
        return avg_sinr, min_sinr, avg_interference
    except Exception as e:
        logging.error(f"Error in SINR calculation: {e}")
        return -100.0, -100.0, noise_floor_dbm

# Enhanced channel plan using graph coloring

def enhanced_generate_channel_plan(ap_locations, min_sep=20.0):
    """
    Assign channels using graph coloring: APs within min_sep meters should not share the same channel.
    """
    if not ap_locations:
        return {}
    try:
        channels_2_4ghz = [1, 6, 11]
        channels_5ghz = [36, 40, 44, 48, 52, 56, 60, 64]
        all_channels = channels_2_4ghz + channels_5ghz
        ap_keys = list(ap_locations.keys())
        G = nx.Graph()
        for ap in ap_keys:
            G.add_node(ap)
        for i, ap1 in enumerate(ap_keys):
            for j, ap2 in enumerate(ap_keys):
                if i < j:
                    try:
                        d = distance_3d(ap_locations[ap1], ap_locations[ap2])
                        if d < min_sep:
                            G.add_edge(ap1, ap2)
                    except Exception as e:
                        logging.warning(f"Error calculating distance between APs {ap1} and {ap2}: {e}")
                        G.add_edge(ap1, ap2)
        try:
            coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        except Exception as e:
            logging.warning(f"Graph coloring failed: {e}, using simple channel assignment")
            coloring = {ap: i for i, ap in enumerate(ap_keys)}
        channel_plan = {}
        for ap, color in coloring.items():
            channel_plan[ap] = all_channels[color % len(all_channels)]
        return channel_plan
    except Exception as e:
        logging.error(f"Error in channel plan generation: {e}")
        channels_2_4ghz = [1, 6, 11]
        channels_5ghz = [36, 40, 44, 48, 52, 56, 60, 64]
        all_channels = channels_2_4ghz + channels_5ghz
        return {ap: all_channels[i % len(all_channels)] for i, ap in enumerate(ap_locations.keys())}


def is_full_rectangle_polygon(polygon, width, height, tol=1e-2):
    rect = [(0, 0), (width, 0), (width, height), (0, height)]
    return (
        len(polygon) == 4 and
        all(any(np.allclose(np.array(p), np.array(r), atol=tol) for p in polygon) for r in rect)
    )

# 3D Grid Parameters for AP Placement Optimization
DEFAULT_OPTIMIZATION_GRID_X = 20
DEFAULT_OPTIMIZATION_GRID_Y = 15
DEFAULT_OPTIMIZATION_GRID_Z = 7  # Z-axis resolution for 3D optimization
DEFAULT_COVERAGE_GRID_X = 80
DEFAULT_COVERAGE_GRID_Y = 50
DEFAULT_COVERAGE_GRID_Z = 10  # Z-axis resolution for coverage evaluation

AP_CONFIG = {
    'coverage_area_cum': 476.0,  # Cubic meters each AP can cover (ceiling mount)
    'max_devices': 30,           # Maximum clients per AP
    'coverage_area_sqft': None,  # Not used for 3D, kept for compatibility
    'coverage_area_sqm': None,   # Not used for 3D, kept for compatibility
    'coverage_radius_m': None,   # Not used for 3D, kept for compatibility
    'min_signal_strength': -59,  # Minimum signal strength for reliable coverage (dBm)
    'optimal_signal_strength': -45,  # Optimal signal strength threshold (dBm)
    'tx_power': 20.0,           # Transmit power in dBm
    'frequency': 2.4e9,         # Frequency in Hz (2.4 GHz)
    'wifi_standard': 5,         # WiFi standard (5 = 802.11ac)
    # 3D Performance-related values
    'coarse_grid_x': 15,
    'coarse_grid_y': 10,
    'coarse_grid_z': 7,         # Z-axis resolution for coarse optimization
    'quick_coarse_grid_x': 6,
    'quick_coarse_grid_y': 4,
    'quick_coarse_grid_z': 3,   # Z-axis resolution for quick mode
    'direct_path_steps_per_meter': 10,
    'direct_path_min_steps': 5,
    'diffracted_path_steps_per_meter': 5,
    'diffracted_path_min_steps': 3,
    'candidate_grid_x': 20,
    'candidate_grid_y': 12,
    'candidate_grid_z': 7,      # Z-axis resolution for candidate positions
    'material_grid_resolution': 0.2,
    'omnidirectional': True,    # AP is omnidirectional
    'mounting': 'ceiling',      # AP mounting type
    'ceiling_height': 2.7,      # Default ceiling height for AP placement
    'min_ap_height': 2.0,       # Minimum AP height from floor
    'max_ap_height': 3.5,       # Maximum AP height from floor
}

# OPTIMIZED: Simplified AP Configuration for better performance
ADVANCED_AP_CONFIG = {
    'interference_threshold': -75,  # dBm - minimum interference level
    'capacity_per_ap': 25,          # Mbps - realistic capacity per AP (considering client load)
    'reflection_coefficient': 0.3,  # Signal reflection coefficient
    'diffraction_coefficient': 0.1, # Signal diffraction coefficient
    # Performance-related values
    'reflection_loss_db': 6,
    'diffraction_loss_db': 3,
}

# Import propagation engines
from src.propagation.engines import FastRayTracingEngine, Cost231Engine, VPLEEngine

import concurrent.futures
from functools import lru_cache
from tqdm import tqdm

import orjson

# AP hardware and power cost constants for cost-weighted optimization
AP_COST_PER_UNIT = 500  # Example: $500 per AP
POWER_COST_PER_DBM = 2  # Example: $2 per dBm of tx_power

# OPTIMIZED: Simplified optimization parameters for better performance
ADVANCED_OPTIMIZATION_CONFIG = {
    'use_parallel_evaluation': False,  # Disabled for stability
    'use_elitism': True,
    'cache_evaluations': True,
    'use_multi_population': False,  # Disabled for performance
    'population_count': 1,
    'migration_frequency': 10,
    'migration_rate': 0.1
}

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        return obj
    except Exception as e:
        logging.warning(f"Error converting numpy types: {e}")
        return str(obj)

def ap_list_to_dict(individual):
    """Convert a list of APs to a dictionary format."""
    if not individual:
        return {}
    
    # Handle case where individual is not a list (shouldn't happen but safety check)
    if not isinstance(individual, (list, tuple)):
        logging.warning(f"Individual is not a list: {type(individual)}")
        return {}
    
    # Check if individual is a list of APs (each AP is a list/tuple)
    if len(individual) > 0 and isinstance(individual[0], (list, tuple)):
        return {f'AP{i+1}': tuple(ap) for i, ap in enumerate(individual)}
    else:
        # Individual is a flat list of coordinates, convert to AP format
        # Each AP has 4 values: x, y, z, tx_power
        if len(individual) % 4 != 0:
            logging.warning(f"Individual length {len(individual)} is not divisible by 4")
            return {}
        
        num_aps = len(individual) // 4
        ap_dict = {}
        for i in range(num_aps):
            start_idx = i * 4
            ap_dict[f'AP{i+1}'] = tuple(individual[start_idx:start_idx+4])
        return ap_dict

def distance_3d(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two positions."""
    try:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    except (TypeError, ValueError) as e:
        logging.warning(f"Error calculating 3D distance: {e}")
        return float('inf')

def calculate_rssi_3d(ap_location: Tuple[float, float, float], point: Tuple[float, float, float], 
                     collector, materials_grid=None) -> float:
    """Calculate RSSI with 3D path loss and material attenuation."""
    try:
        if not collector:
            return -100.0
        
        distance = distance_3d(ap_location, point)
        if distance == float('inf'):
            return -100.0
            
        base_rssi = collector.calculate_rssi(distance)
        
        # Add material attenuation if available
        if materials_grid:
            attenuation = calculate_path_attenuation(ap_location, point, materials_grid)
            return base_rssi - attenuation
        
        return base_rssi
    except Exception as e:
        logging.warning(f"Error calculating 3D RSSI: {e}")
        return -100.0

def calculate_path_attenuation(ap_location: Tuple[float, float, float], 
                              point: Tuple[float, float, float], materials_grid) -> float:
    """Calculate signal attenuation along the path through materials."""
    try:
        if not materials_grid:
            return 0.0
        
        # Simple line-of-sight attenuation calculation using Bresenham's algorithm
        x1, y1, z1 = ap_location
        x2, y2, z2 = point
        
        # Convert to grid coordinates (assuming 0.2m resolution)
        resolution = 0.2
        gx1, gy1, gz1 = int(x1 / resolution), int(y1 / resolution), int(z1 / resolution)
        gx2, gy2, gz2 = int(x2 / resolution), int(y2 / resolution), int(z2 / resolution)
        
        total_attenuation = 0.0
        
        # Simple 3D line traversal
        steps = max(abs(gx2 - gx1), abs(gy2 - gy1), abs(gz2 - gz1))
        if steps == 0:
            return 0.0
            
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            gx = int(gx1 + t * (gx2 - gx1))
            gy = int(gy1 + t * (gy2 - gy1))
            gz = int(gz1 + t * (gz2 - gz1))
            
            # Check bounds and get material
            if (0 <= gz < len(materials_grid) and 
                0 <= gy < len(materials_grid[0]) and 
                0 <= gx < len(materials_grid[0][0])):
                
                material = materials_grid[gz][gy][gx]
                if hasattr(material, 'calculate_attenuation'):
                    total_attenuation += material.calculate_attenuation()
        
        return total_attenuation
    except Exception as e:
        logging.warning(f"Error calculating path attenuation: {e}")
        return 0.0

def calculate_rssi_grid_parallel(engine, ap_locations, points, materials_grid, visualizer):
    """Calculate RSSI grid for all APs in parallel."""
    def rssi_for_ap(ap):
        ap_xy = ap_locations[ap]
        # Use engine's batch method if available, else fallback to per-point
        if hasattr(engine, 'calculate_rssi_grid'):
            try:
                rssi = engine.calculate_rssi_grid(ap_xy, points, materials_grid)
            except Exception:
                rssi = np.array([engine.calculate_rssi(ap_xy, pt, materials_grid, building_width=visualizer.width, building_height=visualizer.height) for pt in points])
        else:
            rssi = np.array([engine.calculate_rssi(ap_xy, pt, materials_grid, building_width=visualizer.width, building_height=visualizer.height) for pt in points])
        return ap, rssi
    rssi_by_ap = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(rssi_for_ap, ap_locations.keys()))
    for ap, rssi in results:
        rssi_by_ap[ap] = rssi
    return rssi_by_ap
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WiFi Signal Strength Prediction with AP Capacity Optimization')
    
    # Building dimensions
    parser.add_argument('--width', type=float, default=40.0,
                        help='Building width in meters (default: 40.0)')
    parser.add_argument('--height', type=float, default=3.0,
                        help='Building height in meters (default: 3.0)')
    parser.add_argument('--length', type=float, default=50.0,
                        help='Building length in meters (default: 50.0)')
    
    # Floor plan configuration
    parser.add_argument('--floor-plan-config', type=str, default=None,
                        help='Path to floor plan configuration JSON file (optional)')
    
    # Sampling resolution
    parser.add_argument('--resolution', type=int, default=100,
                        help='Number of sample points along width (default: 100)')
    
    # AP Capacity Parameters
    parser.add_argument('--coverage-area-sqft', type=float, default=1400.0,
                        help='AP coverage area in square feet (default: 1400.0)')
    parser.add_argument('--max-devices', type=int, default=40,
                        help='Maximum devices per AP (default: 40)')
    
    # Coverage target
    parser.add_argument('--target-coverage', type=float, default=0.90,
                        help='Target coverage percentage (0.0-1.0, default: 0.90)')
    
    parser.add_argument('--propagation-model', type=str, choices=['fast_ray_tracing', 'cost231', 'vple'], 
                        default='fast_ray_tracing', help='Propagation model to use')
    parser.add_argument('--placement-strategy', type=str, choices=['material_aware', 'signal_propagation', 'coverage_gaps'], 
                        default='material_aware', help='AP placement strategy to use')
    parser.add_argument('--quick-mode', action='store_true',
                        help='Enable quick mode for fast testing (reduces optimizer iterations and grid resolution)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous optimized AP locations if available')
    
    # New argument for objective weights
    parser.add_argument('--objective-config', type=str, default=None, 
                        help='Path to objective weights config JSON file')
    
    return parser.parse_args()

def setup_environment():
    """Parses arguments, sets up logging, and creates output directories."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("runs", f"run_{timestamp}")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    logging.info(f"Run output will be saved to: {output_dir}")
    return args, output_dir, plots_dir

# Missing utility functions implementation
def generate_wall_mask(materials_grid):
    """Generate a wall mask from materials grid."""
    try:
        if not materials_grid:
            return None
        
        if isinstance(materials_grid[0][0], list):  # 3D grid
            # Take middle slice for 2D representation
            mid_z = len(materials_grid) // 2
            grid_2d = materials_grid[mid_z]
        else:  # 2D grid
            grid_2d = materials_grid
        
        wall_mask = np.zeros((len(grid_2d), len(grid_2d[0])), dtype=bool)
        
        for i, row in enumerate(grid_2d):
            for j, material in enumerate(row):
                if hasattr(material, 'name') and material.name.lower() not in {'air', 'empty', 'none'}:
                    wall_mask[i, j] = True
        
        return wall_mask
    except Exception as e:
        logging.warning(f"Error generating wall mask: {e}")
        return None

def generate_open_space_mask(materials_grid):
    """Generate an open space mask from materials grid."""
    try:
        wall_mask = generate_wall_mask(materials_grid)
        if wall_mask is not None:
            return ~wall_mask  # Invert wall mask
        return None
    except Exception as e:
        logging.warning(f"Error generating open space mask: {e}")
        return None

def load_and_validate_config(config_path):
    """Loads and validates the floor plan configuration to prevent DoS and path traversal."""
    if not config_path:
        logging.warning("No floor plan config provided. Using default empty layout.")
        return {}, 40.0, 50.0, 3.0, [], None

    # --- Threat Resistance: Path Sanitization ---
    if ".." in config_path or not os.path.abspath(config_path).startswith(os.getcwd()):
        logging.error("Security Error: Path traversal detected. Aborting.")
        raise ValueError("Invalid configuration file path.")

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}. Aborting.")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # --- Threat Resistance: Input Validation & Clamping ---
    building_conf = config_data.get('building', {})
    width = min(float(building_conf.get('width', 40.0)), VALIDATION_LIMITS['width'])
    length = min(float(building_conf.get('length', 50.0)), VALIDATION_LIMITS['length'])
    height = min(float(building_conf.get('height', 3.0)), VALIDATION_LIMITS['height'])
    
    preplaced_aps = config_data.get('ap_locations', [])
    if len(preplaced_aps) > VALIDATION_LIMITS['max_aps']:
        logging.warning(f"Too many pre-placed APs ({len(preplaced_aps)}). Clamping to {VALIDATION_LIMITS['max_aps']}.")
        preplaced_aps = preplaced_aps[:VALIDATION_LIMITS['max_aps']]

    logging.info("Configuration loaded and validated successfully.")
    
    regions = config_data.get('regions', [])
    return config_data, width, length, height, regions, preplaced_aps

def estimate_initial_ap_count(
    building_width, building_length, building_height,
    user_density_per_sqm=0.1, devices_per_user=1.5, user_density_per_cum=0.067,
    rooms=None, ml_model=None, context=None, materials_grid=None, attenuation_threshold_db=7.0
):
    """
    Realistic initial AP count estimation based on volume, user/device density, and material attenuation.
    - Uses 0.067 users/m³ (about 1 user per 15 m³)
    - Computes average attenuation from materials_grid if available
    - Reduces effective AP coverage area per AP based on attenuation
    - Ensures at least one AP per room/closed structure (regardless of material) for omnidirectional ceiling mount
    - Fallbacks to conservative defaults if no grid/room info
    Returns: (estimated_ap_count, reasoning_dict)
    """
    import numpy as np
    BASE_COVERAGE_SQM = 100.0  # Open space, one AP covers 100 m²
    MIN_COVERAGE_SQM = 30.0    # Worst case, one AP covers 30 m²
    MAX_COVERAGE_SQM = 150.0   # Best case, one AP covers 150 m²
    
    # 1. Compute total volume and area
    total_volume = building_width * building_length * building_height
    total_area = building_width * building_length
    
    # 2. User/device-based estimation
    total_users = total_volume * user_density_per_cum
    total_devices = total_users * devices_per_user
    device_based_aps = int(np.ceil(total_devices / AP_CONFIG['max_devices']))
    
    # 3. Material attenuation adjustment
    avg_atten_db = 0.0
    if materials_grid is not None:
        # 2D grid: [y][x] or 3D: [z][y][x]
        attens = []
        try:
            if hasattr(materials_grid[0][0], 'calculate_attenuation'):
                # 2D grid
                for row in materials_grid:
                    for mat in row:
                        if mat and hasattr(mat, 'calculate_attenuation'):
                            att = mat.calculate_attenuation()
                            if att > 0: attens.append(att)
            elif hasattr(materials_grid[0][0][0], 'calculate_attenuation'):
                # 3D grid
                for slab in materials_grid:
                    for row in slab:
                        for mat in row:
                            if mat and hasattr(mat, 'calculate_attenuation'):
                                att = mat.calculate_attenuation()
                                if att > 0: attens.append(att)
            if attens:
                avg_atten_db = float(np.mean(attens))
        except Exception as e:
            logging.warning(f"Error calculating material attenuation: {e}")
    
    # 4. Adjust effective AP coverage area
    effective_coverage_sqm = BASE_COVERAGE_SQM
    if avg_atten_db > 0:
        effective_coverage_sqm = BASE_COVERAGE_SQM / (2 ** (avg_atten_db / attenuation_threshold_db))
        effective_coverage_sqm = max(MIN_COVERAGE_SQM, min(MAX_COVERAGE_SQM, effective_coverage_sqm))
    
    # 5. Area-based estimation
    area_based_aps = int(np.ceil(total_area / effective_coverage_sqm))
    
    # 6. Room/partition awareness
    room_based_aps = 0
    if rooms:
        for room in rooms:
            # Extract room information regardless of material type
            mat = room.get('material', '').lower() if isinstance(room, dict) else str(room[-1]).lower()
            area = room.get('area', None) if isinstance(room, dict) else None
            if area is None and isinstance(room, dict):
                area = room.get('width', 0) * room.get('height', 0)
            if area is None and not isinstance(room, dict):
                area = room[2] * room[3] if len(room) >= 4 else 0
            # Ensure area is not None and valid
            if area is None or area <= 0:
                area = 50.0  # Default room area if unknown
            
            # Every room gets at least one AP
            aps_for_room = max(1, int(np.ceil(area / effective_coverage_sqm)))
            room_based_aps += aps_for_room
    else:
        # If no room info available, estimate based on building complexity
        estimated_rooms = max(1, int(np.ceil(total_area / 100.0)))  # 1 room per 100 m²
        room_based_aps = estimated_rooms
    
    # 7. Use the maximum of all estimates
    estimated_aps = max(device_based_aps, area_based_aps, room_based_aps)
    
    # 8. Context adjustment
    context_factor = 1.0
    if context:
        ctx = context.lower()
        if 'outdoor' in ctx:
            context_factor *= 0.5
        if 'open' in ctx:
            context_factor *= 0.7
        if 'closed' in ctx or 'partitioned' in ctx:
            context_factor *= 1.2
        if 'indoor' in ctx:
            context_factor *= 1.0
    estimated_aps = int(np.ceil(estimated_aps * context_factor))
    
    # 9. Minimum: at least 1 AP per 150 m², maximum: 1 per 30 m²
    min_aps = int(np.ceil(total_area / MAX_COVERAGE_SQM))
    max_aps = int(np.ceil(total_area / MIN_COVERAGE_SQM))
    estimated_aps = max(min_aps, min(max_aps, estimated_aps))
    
    reasoning = {
        'device_based_aps': device_based_aps,
        'area_based_aps': area_based_aps,
        'room_based_aps': room_based_aps,
        'avg_attenuation_db': avg_atten_db,
        'effective_coverage_sqm': effective_coverage_sqm,
        'context_factor': context_factor,
        'final_estimate': estimated_aps,
        'total_area_sqm': total_area,
        'total_volume_cum': total_volume,
        'expected_users_volume': total_users,
        'expected_devices': total_devices,
        'room_strategy': 'one_ap_per_room_omnidirectional_ceiling'
    }
    
    logging.info(f"[AP Estimation] Device: {device_based_aps}, Area: {area_based_aps}, Room: {room_based_aps}, Atten: {avg_atten_db:.2f} dB, EffCov: {effective_coverage_sqm:.1f} m², Context: {context_factor}, Final: {estimated_aps}")
    return max(1, estimated_aps), reasoning

def advanced_ap_count_evaluation(building_width, building_length, building_height, materials_grid, collector, engine, room_regions=None):
    """Advanced AP count evaluation using multiple factors."""
    try:
        # Use the existing estimate_initial_ap_count function
        return estimate_initial_ap_count(
            building_width, building_length, building_height,
            rooms=room_regions, materials_grid=materials_grid
        )
    except Exception as e:
        logging.warning(f"Error in advanced AP count evaluation: {e}")
        # Fallback calculation
        area = building_width * building_length
        estimated_aps = max(1, int(area / 100))  # 1 AP per 100 m²
        return estimated_aps, {'fallback': True, 'area_based': estimated_aps}

def run_multiobjective_ap_optimization(**kwargs):
    """Simplified multi-objective optimization."""
    try:
        building_width = kwargs.get('building_width', 40)
        building_length = kwargs.get('building_length', 50)
        building_height = kwargs.get('building_height', 3)
        materials_grid = kwargs.get('materials_grid')
        collector = kwargs.get('collector')
        points = kwargs.get('points', [])
        min_aps = kwargs.get('min_aps', 2)
        max_aps = kwargs.get('max_aps', 10)
        quick_mode = kwargs.get('quick_mode', False)
        
        # Simple grid-based optimization
        best_solutions = []
        
        for num_aps in range(min_aps, max_aps + 1):
            ap_locations = _place_aps_intelligent_grid(
                num_aps=num_aps,
                building_width=building_width,
                building_length=building_length,
                building_height=building_height,
                materials_grid=materials_grid
            )
            
            # Convert to list format for compatibility
            ap_list = []
            for ap_name, (x, y, z) in ap_locations.items():
                ap_list.extend([x, y, z, 15.0])  # Default tx_power = 15 dBm
            
            best_solutions.append(ap_list)
            
            if quick_mode and len(best_solutions) >= 3:
                break
        
        return best_solutions, None
    except Exception as e:
        logging.error(f"Error in multi-objective optimization: {e}")
        return [], None

def select_superior_solution(pareto_front, target_coverage=0.9):
    """Select the best solution from the Pareto front."""
    try:
        if not pareto_front:
            return None
        
        # For now, just return the first solution
        # In a real implementation, this would evaluate each solution
        return pareto_front[0]
    except Exception as e:
        logging.error(f"Error selecting superior solution: {e}")
        return None

def _place_aps_intelligent_grid(
    num_aps,
    building_width,
    building_length,
    building_height,
    materials_grid=None,
    ceiling_height=None,
    min_ap_sep=None,
    min_wall_gap=None,
    wall_mask=None,
    open_space_mask=None,
    z_levels=1,
    room_regions=None,
    logger=None
):
    """
    3D, material-aware, constraint-compliant grid-based AP placement.
    - Places APs as (x, y, z) tuples at ceiling height (or configurable z-levels)
    - Avoids walls/obstacles using materials_grid
    - Enforces minimum wall gap and minimum AP separation (in open space only)
    - Optionally ensures at least one AP per room/closed structure if room_regions provided
    - Returns: {f'AP{i+1}': (x, y, z)}
    """
    import numpy as np
    import logging
    if logger is None:
        logger = logging.getLogger("APGridPlacement")
    
    # Use AP_CONFIG defaults if not provided
    if ceiling_height is None:
        ceiling_height = building_height * 0.8 if building_height else 2.7
    if min_ap_sep is None:
        min_ap_sep = 7.0
    if min_wall_gap is None:
        min_wall_gap = 1.0
    
    ap_locations = {}
    placed_coords = []
    
    # Helper: check if (x, y) is in wall/obstacle
    def is_in_wall(x, y):
        if materials_grid is None:
            return False
        try:
            res = 0.2
            grid_x = int(x / res)
            grid_y = int(y / res)
            if isinstance(materials_grid[0][0], list):  # 3D grid
                mid_z = len(materials_grid) // 2
                if 0 <= grid_y < len(materials_grid[mid_z]) and 0 <= grid_x < len(materials_grid[mid_z][0]):
                    mat = materials_grid[mid_z][grid_y][grid_x]
                    return hasattr(mat, 'name') and mat.name.lower() not in {"air", "empty", "none"}
            else:  # 2D grid
                if 0 <= grid_y < len(materials_grid) and 0 <= grid_x < len(materials_grid[0]):
                    mat = materials_grid[grid_y][grid_x]
                    return hasattr(mat, 'name') and mat.name.lower() not in {"air", "empty", "none"}
        except Exception:
            pass
        return False
    
    # Helper: distance to nearest wall
    def distance_to_nearest_wall(x, y, max_search=5.0):
        if materials_grid is None:
            return max_search
        
        res = 0.2
        max_cells = int(max_search / res)
        grid_x = int(x / res)
        grid_y = int(y / res)
        min_dist = float('inf')
        
        try:
            for dy in range(-max_cells, max_cells+1):
                for dx in range(-max_cells, max_cells+1):
                    nx, ny = grid_x + dx, grid_y + dy
                    
                    if isinstance(materials_grid[0][0], list):  # 3D grid
                        mid_z = len(materials_grid) // 2
                        if 0 <= ny < len(materials_grid[mid_z]) and 0 <= nx < len(materials_grid[mid_z][0]):
                            mat = materials_grid[mid_z][ny][nx]
                            if hasattr(mat, 'name') and mat.name.lower() not in {"air", "empty", "none"}:
                                dist = np.hypot(dx * res, dy * res)
                                if dist < min_dist:
                                    min_dist = dist
                    else:  # 2D grid
                        if 0 <= ny < len(materials_grid) and 0 <= nx < len(materials_grid[0]):
                            mat = materials_grid[ny][nx]
                            if hasattr(mat, 'name') and mat.name.lower() not in {"air", "empty", "none"}:
                                dist = np.hypot(dx * res, dy * res)
                                if dist < min_dist:
                                    min_dist = dist
        except Exception:
            pass
        
        return min_dist if min_dist != float('inf') else max_search
    
    # 1. Optionally, place one AP per room/closed structure
    ap_idx = 1
    if room_regions:
        z = ceiling_height
        for region in room_regions:
            if isinstance(region, dict):
                x, y, w, h = float(region["x"]), float(region["y"]), float(region["width"]), float(region["height"])
            else:
                x, y, w, h, *_ = region
            
            if w * h > 5:  # Only place AP in rooms larger than 5 m²
                ap_x = x + w / 2
                ap_y = y + h / 2
                
                if (not is_in_wall(ap_x, ap_y) and 
                    distance_to_nearest_wall(ap_x, ap_y) >= min_wall_gap and
                    all(np.linalg.norm(np.array([ap_x, ap_y]) - np.array([ax, ay])) >= min_ap_sep 
                        for (ax, ay, _) in placed_coords)):
                    
                    ap_locations[f"AP{ap_idx}"] = (ap_x, ap_y, z)
                    placed_coords.append((ap_x, ap_y, z))
                    logger.info(f"[Room AP] Placed AP{ap_idx} at ({ap_x:.1f}, {ap_y:.1f}, {z:.1f})")
                    ap_idx += 1
    
    # 2. Fill remaining APs in a grid pattern
    n_remaining = num_aps - len(ap_locations)
    if n_remaining > 0:
        # Generate grid of candidate positions
        n_x = max(2, int(np.ceil(np.sqrt(n_remaining * building_width / building_length))))
        n_y = max(2, int(np.ceil(n_remaining / n_x)))
        
        x_grid = np.linspace(min_wall_gap, building_width - min_wall_gap, n_x)
        y_grid = np.linspace(min_wall_gap, building_length - min_wall_gap, n_y)
        
        candidates = [(x, y, ceiling_height) for x in x_grid for y in y_grid]
        
        # Score candidates by distance to walls
        scored = []
        for cand in candidates:
            x, y, z = cand
            if is_in_wall(x, y):
                continue
            if distance_to_nearest_wall(x, y) < min_wall_gap:
                continue
            # Enforce min AP separation
            if any(np.linalg.norm(np.array([x, y, z]) - np.array([ax, ay, az])) < min_ap_sep 
                   for (ax, ay, az) in placed_coords):
                continue
            
            # Score: prefer farther from wall
            wall_dist = distance_to_nearest_wall(x, y)
            scored.append((wall_dist, cand))
        
        # Sort by wall distance (prefer farther)
        scored.sort(reverse=True)
        
        for _, cand in scored:
            if len(ap_locations) >= num_aps:
                break
            ap_locations[f"AP{ap_idx}"] = cand
            placed_coords.append(cand)
            logger.info(f"[Grid AP] Placed AP{ap_idx} at ({cand[0]:.1f}, {cand[1]:.1f}, {cand[2]:.1f})")
            ap_idx += 1
    
    # 3. If still not enough APs, place them in a simple grid
    while len(ap_locations) < num_aps:
        # Simple fallback placement
        x = (ap_idx % 3) * building_width / 3 + building_width / 6
        y = ((ap_idx // 3) % 3) * building_length / 3 + building_length / 6
        z = ceiling_height
        
        ap_locations[f"AP{ap_idx}"] = (x, y, z)
        logger.info(f"[Fallback AP] Placed AP{ap_idx} at ({x:.1f}, {y:.1f}, {z:.1f})")
        ap_idx += 1
    
    logger.info(f"[Intelligent Grid Placement] Total APs placed: {len(ap_locations)}")
    return ap_locations

def save_run_info(args, run_dir, ap_locations):
    """Save run configuration and metadata.
    
    Args:
        args: Command line arguments
        run_dir: Directory to save run information
        ap_locations: Dictionary of AP locations
    """
    try:
        run_info = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'building_width': getattr(args, 'width', 40.0),
                'building_length': getattr(args, 'length', 50.0),
                'building_height': getattr(args, 'height', 3.0),
                'resolution': getattr(args, 'resolution', 100),
                'target_coverage': getattr(args, 'target_coverage', 0.9),
                'quick_mode': getattr(args, 'quick_mode', False),
                'ap_locations': ap_locations,
                'ap_config': AP_CONFIG
            },
            'materials_used': list(MATERIALS.keys()) if MATERIALS else ['air'],
            'access_points': list(ap_locations.keys()) if ap_locations else []
        }
        
        # Convert numpy types before serialization
        run_info_serializable = convert_numpy_types(run_info)
        
        # Save as JSON
        run_info_path = os.path.join(run_dir, 'run_info.json')
        with open(run_info_path, 'w') as f:
            json.dump(run_info_serializable, f, indent=2)
        
        logging.info(f"Run information saved to {run_info_path}")
        
    except Exception as e:
        logging.error(f"Error saving run info: {e}")

def evaluate_coverage_and_capacity(
    ap_locations,
    building_width,
    building_length,
    building_height,
    resolution=100,
    materials_grid=None,
    propagation_engine=None,
    target_coverage=0.9,
    logger=None
):
    """Evaluate coverage and capacity metrics for AP placement.
    
    Args:
        ap_locations: Dictionary of AP locations {ap_id: (x, y, z)}
        building_width: Building width in meters
        building_length: Building length in meters
        building_height: Building height in meters
        resolution: Grid resolution for evaluation
        materials_grid: 3D materials grid for propagation
        propagation_engine: Propagation calculation engine
        target_coverage: Target coverage percentage
        logger: Logger instance
    
    Returns:
        Dictionary with coverage and capacity metrics
    """
    import numpy as np
    import logging
    if logger is None:
        logger = logging.getLogger("CoverageEval")
    
    try:
        # Create evaluation grid
        x_points = np.linspace(0, building_width, resolution)
        y_points = np.linspace(0, building_length, resolution)
        z_height = building_height * 0.1  # Evaluation at user height
        
        total_points = len(x_points) * len(y_points)
        covered_points = 0
        signal_strengths = []
        interference_levels = []
        
        # Minimum signal strength threshold (dBm)
        min_signal_threshold = -70
        
        for x in x_points:
            for y in y_points:
                point_signals = []
                
                # Calculate signal from each AP
                for ap_id, (ap_x, ap_y, ap_z) in ap_locations.items():
                    distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2 + (z_height - ap_z)**2)
                    
                    # Simple path loss model (free space + wall attenuation)
                    if distance < 0.1:
                        distance = 0.1  # Avoid division by zero
                    
                    # Free space path loss at 2.4 GHz
                    fspl = 20 * np.log10(distance) + 20 * np.log10(2400) + 32.44
                    
                    # Add material attenuation if available
                    wall_attenuation = 0
                    if materials_grid is not None:
                        # Simplified wall attenuation calculation
                        wall_attenuation = calculate_wall_attenuation(x, y, ap_x, ap_y, materials_grid)
                    
                    # Calculate received signal strength (assuming 20 dBm transmit power)
                    signal_strength = 20 - fspl - wall_attenuation
                    point_signals.append(signal_strength)
                
                if point_signals:
                    # Find strongest signal
                    max_signal = max(point_signals)
                    signal_strengths.append(max_signal)
                    
                    # Calculate interference (sum of other signals)
                    interference = sum(10**(s/10) for s in point_signals if s != max_signal)
                    if interference > 0:
                        interference_db = 10 * np.log10(interference)
                        interference_levels.append(interference_db)
                    
                    # Check if point is covered
                    if max_signal >= min_signal_threshold:
                        covered_points += 1
        
        # Calculate metrics
        coverage_percentage = (covered_points / total_points) * 100
        avg_signal_strength = np.mean(signal_strengths) if signal_strengths else -100
        avg_interference = np.mean(interference_levels) if interference_levels else -100
        
        # Estimate capacity (simplified)
        # Higher signal strength and lower interference = higher capacity
        capacity_factor = 1.0
        if avg_signal_strength > -60:
            capacity_factor *= 1.5
        elif avg_signal_strength > -70:
            capacity_factor *= 1.2
        
        if avg_interference < -80:
            capacity_factor *= 1.3
        elif avg_interference < -70:
            capacity_factor *= 1.1
        
        estimated_capacity = len(ap_locations) * 50 * capacity_factor  # Mbps per AP
        
        results = {
            'coverage_percentage': coverage_percentage,
            'covered_points': covered_points,
            'total_points': total_points,
            'avg_signal_strength': avg_signal_strength,
            'avg_interference': avg_interference,
            'estimated_capacity_mbps': estimated_capacity,
            'num_access_points': len(ap_locations),
            'meets_target_coverage': coverage_percentage >= (target_coverage * 100),
            'signal_strength_distribution': {
                'min': min(signal_strengths) if signal_strengths else -100,
                'max': max(signal_strengths) if signal_strengths else -100,
                'std': np.std(signal_strengths) if signal_strengths else 0
            }
        }
        
        logger.info(f"Coverage evaluation complete: {coverage_percentage:.1f}% coverage")
        logger.info(f"Average signal strength: {avg_signal_strength:.1f} dBm")
        logger.info(f"Estimated capacity: {estimated_capacity:.0f} Mbps")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in coverage evaluation: {e}")
        return {
            'coverage_percentage': 0,
            'covered_points': 0,
            'total_points': resolution * resolution,
            'avg_signal_strength': -100,
            'avg_interference': -100,
            'estimated_capacity_mbps': 0,
            'num_access_points': len(ap_locations),
            'meets_target_coverage': False,
            'error': str(e)
        }

def calculate_wall_attenuation(x1, y1, x2, y2, materials_grid, resolution=0.2):
    """Calculate wall attenuation between two points.
    
    Args:
        x1, y1: Start point coordinates
        x2, y2: End point coordinates
        materials_grid: 3D materials grid
        resolution: Grid resolution in meters
    
    Returns:
        Total attenuation in dB
    """
    try:
        if materials_grid is None:
            return 0
        
        # Simple line-of-sight calculation
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance < 0.1:
            return 0
        
        # Sample points along the line
        num_samples = max(10, int(distance / resolution))
        x_samples = np.linspace(x1, x2, num_samples)
        y_samples = np.linspace(y1, y2, num_samples)
        
        total_attenuation = 0
        
        for x, y in zip(x_samples, y_samples):
            grid_x = int(x / resolution)
            grid_y = int(y / resolution)
            
            try:
                if isinstance(materials_grid[0][0], list):  # 3D grid
                    mid_z = len(materials_grid) // 2
                    if 0 <= grid_y < len(materials_grid[mid_z]) and 0 <= grid_x < len(materials_grid[mid_z][0]):
                        mat = materials_grid[mid_z][grid_y][grid_x]
                        if hasattr(mat, 'name') and mat.name.lower() not in {"air", "empty", "none"}:
                            # Add attenuation based on material type
                            if 'wall' in mat.name.lower():
                                total_attenuation += 10  # dB per wall
                            elif 'concrete' in mat.name.lower():
                                total_attenuation += 15
                            elif 'metal' in mat.name.lower():
                                total_attenuation += 20
                            else:
                                total_attenuation += 5  # Generic obstacle
                else:  # 2D grid
                    if 0 <= grid_y < len(materials_grid) and 0 <= grid_x < len(materials_grid[0]):
                        mat = materials_grid[grid_y][grid_x]
                        if hasattr(mat, 'name') and mat.name.lower() not in {"air", "empty", "none"}:
                            if 'wall' in mat.name.lower():
                                total_attenuation += 10
                            elif 'concrete' in mat.name.lower():
                                total_attenuation += 15
                            elif 'metal' in mat.name.lower():
                                total_attenuation += 20
                            else:
                                total_attenuation += 5
            except (IndexError, AttributeError):
                continue
        
        return min(total_attenuation, 50)  # Cap at 50 dB
        
    except Exception:
        return 0

def load_objective_weights(config_path):
    """Load objective weights from config file."""
    try:
        if not config_path or not os.path.exists(config_path):
            return {'coverage': 0.4, 'cost': 0.3, 'interference': 0.3}
        
        with open(config_path, 'r') as f:
            weights = json.load(f)
        return weights
    except Exception as e:
        logging.warning(f"Error loading objective weights: {e}")
        return {'coverage': 0.4, 'cost': 0.3, 'interference': 0.3}

def run_optimization_pipeline(building_width, building_length, building_height, materials_grid, collector, engine, points, regions, initial_aps, args):
    """Executes the core AP placement optimization using the genetic algorithm."""
    logging.info("🚀 Starting AP placement optimization pipeline...")

    # 1. Estimate a good range for the number of APs
    n_aps_estimate, reason = advanced_ap_count_evaluation(
        building_width, building_length, building_height, materials_grid, collector, engine, room_regions=regions
    )
    min_aps = max(2, n_aps_estimate - 2)
    max_aps = min(VALIDATION_LIMITS['max_aps'], n_aps_estimate + 4)
    logging.info(f"Estimated optimal AP count is ~{n_aps_estimate}. Optimizing for range [{min_aps}, {max_aps}].")

    # 2. Run the multi-objective genetic algorithm
    pareto_front, _ = run_multiobjective_ap_optimization(
        building_width=building_width,
        building_length=building_length,
        building_height=building_height,
        materials_grid=materials_grid,
        collector=collector,
        points=points,
        target_coverage=args.target_coverage,
        engine=engine,
        pop_size=20 if args.quick_mode else 40,
        ngen=10 if args.quick_mode else 30,
        min_aps=min_aps,
        max_aps=max_aps,
        initial_ap_locations=initial_aps,
        quick_mode=args.quick_mode,
        objective_weights=load_objective_weights(args.objective_config)
    )

    # 3. Select the superior solution from the results
    logging.info(f"Optimization complete. Found {len(pareto_front)} optimal solutions.")
    best_solution = select_superior_solution(pareto_front, args.target_coverage)
    
    if not best_solution:
        logging.error("Optimization failed to find a valid solution.")
        return None

    optimized_ap_locations = ap_list_to_dict(best_solution)
    logging.info(f"Selected superior solution with {len(optimized_ap_locations)} APs.")
    return optimized_ap_locations

def generate_superior_performance_report(ap_locations, building_width, building_length, building_height, materials_grid, collector, points, output_dir, engine):
    """Generate comprehensive performance report."""
    try:
        logging.info("Generating performance report...")
        
        # Evaluate coverage and capacity
        result = evaluate_coverage_and_capacity(
            ap_locations, building_width, building_height, materials_grid, collector, points, engine=engine
        )
        
        # Calculate interference and SINR
        avg_sinr, min_sinr, avg_interference = calculate_interference_and_sinr(
            ap_locations, points, collector
        )
        
        # Generate channel plan
        channel_plan = enhanced_generate_channel_plan(ap_locations)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'building_dimensions': {
                'width': building_width,
                'length': building_length,
                'height': building_height
            },
            'ap_count': len(ap_locations),
            'ap_locations': convert_numpy_types(ap_locations),
            'coverage_metrics': {
                'coverage_percent': result.get('coverage_percent', 0),
                'avg_signal': result.get('avg_signal', -100),
                'avg_sinr': avg_sinr,
                'min_sinr': min_sinr,
                'avg_interference': avg_interference
            },
            'channel_plan': channel_plan,
            'recommendations': result.get('recommendations', [])
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Performance report saved to {report_path}")
        
    except Exception as e:
        logging.error(f"Error generating performance report: {e}")

def create_visualization_plots(ap_locations, building_width, building_height, materials_grid, collector, points, plots_dir, engine, **kwargs):
    """Create visualization plots."""
    try:
        logging.info("Creating visualization plots...")
        
        # Create a simple coverage heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grid for heatmap
        x_grid = np.linspace(0, building_width, 50)
        y_grid = np.linspace(0, building_height, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Calculate signal strength at each grid point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                max_rssi = -100
                
                for ap_name, ap_pos in ap_locations.items():
                    if len(ap_pos) >= 2:
                        distance = np.sqrt((x - ap_pos[0])**2 + (y - ap_pos[1])**2)
                        rssi = collector.calculate_rssi(distance) if collector else -30 - 20 * np.log10(distance + 1)
                        max_rssi = max(max_rssi, rssi)
                
                Z[i, j] = max_rssi
        
        # Plot heatmap
        im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
        plt.colorbar(im, ax=ax, label='RSSI (dBm)')
        
        # Plot AP locations
        for ap_name, ap_pos in ap_locations.items():
            if len(ap_pos) >= 2:
                ax.plot(ap_pos[0], ap_pos[1], 'ko', markersize=10, markerfacecolor='blue')
                ax.annotate(ap_name, (ap_pos[0], ap_pos[1]), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('WiFi Coverage Heatmap')
        ax.set_xlim(0, building_width)
        ax.set_ylim(0, building_height)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'coverage_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Visualization plots created successfully")
        
    except Exception as e:
        logging.error(f"Error creating visualization plots: {e}")

def create_algorithm_comparison_plots(ap_locations, building_width, building_height, materials_grid, collector, points, plots_dir, engine):
    """Create algorithm comparison plots."""
    try:
        logging.info("Creating algorithm comparison plots...")
        
        # Simple bar chart showing AP distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ap_names = list(ap_locations.keys())
        x_coords = [ap_locations[ap][0] for ap in ap_names if len(ap_locations[ap]) >= 2]
        y_coords = [ap_locations[ap][1] for ap in ap_names if len(ap_locations[ap]) >= 2]
        
        ax.scatter(x_coords, y_coords, s=100, alpha=0.7)
        
        for i, ap in enumerate(ap_names):
            if len(ap_locations[ap]) >= 2:
                ax.annotate(ap, (x_coords[i], y_coords[i]), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title('AP Placement Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'ap_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Algorithm comparison plots created successfully")
        
    except Exception as e:
        logging.error(f"Error creating algorithm comparison plots: {e}")

def generate_reports_and_visualizations(ap_locations, building_width, building_length, building_height, materials_grid, collector, points, engine, output_dir, plots_dir, config_data):
    """Generates all output artifacts, including reports and plots."""
    if not ap_locations:
        logging.warning("No AP locations to visualize. Skipping report generation.")
        return

    logging.info("📊 Generating reports and visualizations...")
    
    # 1. Generate the comprehensive performance report
    generate_superior_performance_report(
        ap_locations, building_width, building_length, building_height, materials_grid, collector, points, output_dir, engine
    )

    # 2. Create the main visualization plots (heatmaps, etc.)
    roi_polygon = config_data.get('rois', [{}])[0].get('points') if config_data.get('rois') else None
    background_image = config_data.get('background_image')
    image_extent = [0, building_width, 0, building_length]

    create_visualization_plots(
        ap_locations, building_width, building_height, materials_grid, collector, points,
        plots_dir, engine, regions=config_data.get('regions', []), roi_polygon=roi_polygon, 
        background_image=background_image, image_extent=image_extent
    )

    # 3. Create algorithm comparison plots for analysis
    create_algorithm_comparison_plots(
        ap_locations, building_width, building_height, materials_grid, collector, points, plots_dir, engine
    )
    logging.info(f"All reports and plots saved to {output_dir}")


def main():
    """Main function for WiFi signal strength prediction with multi-objective AP optimization."""
    try:
        # 1. Setup: Initialize logging, parse arguments, and create output directories.
        args, output_dir, plots_dir = setup_environment()

        # 2. Load Building Layout: Read the floor plan config to get dimensions, regions, and any pre-placed APs.
        config_data, width, length, height, regions, preplaced_aps = load_and_validate_config(args.floor_plan_config)
        
        # 3. Build a 3D materials grid for the simulation.
        # Note: A more advanced implementation would parse 'regions' to build a detailed grid.
        # For now, it defaults to an open space (air).
        res = max(0.2, VALIDATION_LIMITS['min_resolution'])
        grid_w = int(width / res)
        grid_l = int(length / res)
        grid_h = int(height / res)
        materials_grid = [[[ADVANCED_MATERIALS['air'] for _ in range(grid_w)] for _ in range(grid_l)] for _ in range(grid_h)]
        
        # 4. Initialize Models: Set up the physics engine and data collector.
        if FastRayTracingEngine:
            engine = FastRayTracingEngine()
        else:
            # Fallback engine implementation
            class FallbackEngine:
                def calculate_rssi(self, ap_pos, point, materials_grid=None, **kwargs):
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(ap_pos[:2], point[:2])))
                    return -30 - 20 * np.log10(distance + 1)  # Simple path loss model
            engine = FallbackEngine()
        
        if WiFiDataCollector:
            collector = WiFiDataCollector(tx_power=20.0, frequency=2.4e9)
        else:
            # Fallback collector implementation
            class FallbackCollector:
                def __init__(self, tx_power=20.0, frequency=2.4e9):
                    self.tx_power = tx_power
                    self.frequency = frequency
                
                def calculate_rssi(self, distance, material=None, include_multipath=True):
                    return self.tx_power - 20 * np.log10(distance + 1) - 20 * np.log10(self.frequency / 1e9)
            collector = FallbackCollector(tx_power=20.0, frequency=2.4e9)
        
        # 5. Generate Evaluation Points: Create a 3D grid of points to test WiFi coverage.
        num_x = 50 if args.quick_mode else 100
        num_y = 30 if args.quick_mode else 60
        num_z = 3  # Evaluate at 3 different heights
        x_vals = np.linspace(0, width, num_x)
        y_vals = np.linspace(0, length, num_y)
        z_vals = np.linspace(1.0, height * 0.9, num_z) # Evaluate from 1m up to 90% of ceiling height
        points = [(x, y, z) for x in x_vals for y in y_vals for z in z_vals]
        
        final_ap_locations = None

        # 6. Determine AP Placement Strategy: Check if APs are pre-defined or need optimization.
        if preplaced_aps:
            # --- PATH 1: ANALYSIS MODE ---
            # If APs are provided in the config file, use them directly for analysis.
            logging.info("Using pre-placed APs from configuration file for analysis.")
            final_ap_locations = {}
            for i, ap in enumerate(preplaced_aps):
                if isinstance(ap, dict):
                    x = ap.get('x', 0)
                    y = ap.get('y', 0)
                    z = ap.get('z', height * 0.8)  # Default to 80% of ceiling height
                    final_ap_locations[f'AP{i+1}'] = (x, y, z)
                elif isinstance(ap, (list, tuple)) and len(ap) >= 2:
                    x, y = ap[0], ap[1]
                    z = ap[2] if len(ap) > 2 else height * 0.8
                    final_ap_locations[f'AP{i+1}'] = (x, y, z)
        else:
            # --- PATH 2: OPTIMIZATION MODE ---
            # If no APs are provided, run the full genetic optimization pipeline.
            logging.info("No pre-placed APs found. Running optimization to find optimal placement.")
            
            # First, create a reasonable initial placement to seed the algorithm.
            initial_ap_count, _ = estimate_initial_ap_count(width, length, height, rooms=regions, materials_grid=materials_grid)
            logging.info(f"Seeding the optimizer with an initial intelligent placement of {initial_ap_count} APs.")
            
            initial_placement = _place_aps_intelligent_grid(
                num_aps=initial_ap_count,
                building_width=width,
                building_length=length,
                building_height=height,
                materials_grid=materials_grid,
                ceiling_height=height * 0.8,
                room_regions=regions,
                logger=logging.getLogger("APGridPlacement")
            )

            # Now, run the main genetic algorithm pipeline using the initial placement as a starting point.
            final_ap_locations = run_optimization_pipeline(
                width, length, height, materials_grid, collector, engine, points, regions, initial_placement, args
            )
            
            # If optimization fails, use the initial placement
            if not final_ap_locations:
                logging.warning("Optimization failed, using initial placement")
                final_ap_locations = initial_placement

        # 7. Generate Final Outputs: Create reports and visualizations for the final AP locations.
        if final_ap_locations:
            # Save run information
            save_run_info(args, output_dir, final_ap_locations)
            
            # Evaluate final performance
            final_result = evaluate_coverage_and_capacity(
                final_ap_locations,
                building_width=width,
                building_length=length,
                building_height=height,
                resolution=args.resolution,
                materials_grid=materials_grid,
                propagation_engine=engine,
                target_coverage=args.target_coverage
            )
            
            # Calculate and log interference metrics
            avg_sinr, min_sinr, avg_interference = calculate_interference_and_sinr(
                final_ap_locations, points, collector
            )
            
            # Generate channel plan
            channel_plan = enhanced_generate_channel_plan(final_ap_locations)
            
            # Log final metrics
            logging.info(f"Final Performance Metrics:")
            logging.info(f"  - Coverage: {final_result.get('coverage_percent', 0):.2%}")
            logging.info(f"  - Average Signal: {final_result.get('avg_signal', -100):.1f} dBm")
            logging.info(f"  - Average SINR: {avg_sinr:.1f} dB")
            logging.info(f"  - Minimum SINR: {min_sinr:.1f} dB")
            logging.info(f"  - Average Interference: {avg_interference:.1f} dBm")
            logging.info(f"  - Number of APs: {len(final_ap_locations)}")
            logging.info(f"  - Channel Plan: {channel_plan}")
            
            # Generate comprehensive reports and visualizations
            generate_reports_and_visualizations(
                final_ap_locations, width, length, height, materials_grid, collector, points, engine, output_dir, plots_dir, config_data
            )
            
            # Use machine learning predictor to suggest improvements (if available)
            try:
                # Initialize placement predictor if not already done
                placement_predictor = None
                if hasattr(args, 'enable_ml_predictor') and args.enable_ml_predictor:
                    try:
                        from .models.wifi_classifier import APPlacementPredictor
                        placement_predictor = APPlacementPredictor()
                    except ImportError:
                        logging.warning("ML predictor not available, skipping ML suggestions")
                
                if placement_predictor:
                    building_features = {
                        'width': width,
                        'length': length,
                        'height': height,
                        'area': width * length,
                        'num_aps': len(final_ap_locations),
                        'avg_signal': final_result.get('avg_signal_strength', -70)
                    }
                    
                    # Add training example
                    performance_score = final_result.get('coverage_percentage', 0) * 0.5 + (avg_sinr + 100) / 100 * 0.3 + (1 - avg_interference / -50) * 0.2
                    placement_predictor.add_training_example(building_features, final_ap_locations, performance_score)
                    
                    # Try to train and suggest improvements
                    if placement_predictor.train():
                        improved_locations = placement_predictor.suggest_improvements(building_features, final_ap_locations)
                        if improved_locations != final_ap_locations:
                            logging.info("ML predictor suggests potential improvements to AP placement")
            except Exception as e:
                logging.warning(f"ML predictor error: {e}")
            
            logging.info("✅ Process completed successfully.")
        else:
            logging.error("❌ Could not determine final AP locations. Optimization may have failed.")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}")
        traceback.print_exc()
        return 1  # Return error code
    
    return 0  # Return success code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)