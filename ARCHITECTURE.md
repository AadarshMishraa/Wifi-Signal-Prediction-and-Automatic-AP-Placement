# Project Architecture

## Overview

WiFi Signal Prediction and AP Placement is a comprehensive system for optimizing wireless network infrastructure using physics-based modeling and machine learning.

## Directory Structure

```
Wifi-Signal-Prediction-and-Automatic-AP-Placement/
├── src/                          # Core source code
│   ├── data_collection/          # WiFi data collection and simulation
│   │   ├── collector.py
│   │   └── wifi_data_collector.py
│   ├── models/                   # Machine learning models
│   │   ├── wifi_classifier.py
│   │   └── wifi_models.py
│   ├── physics/                  # Physics-based signal modeling
│   │   ├── adaptive_voxel_system.py
│   │   └── materials.py
│   ├── preprocessing/            # Data preprocessing
│   │   ├── data_augmentation.py
│   │   ├── feature_engineering.py
│   │   └── preprocessor.py
│   ├── propagation/              # Signal propagation engines
│   │   └── engines.py
│   ├── utils/                    # Utility functions
│   │   ├── error_handling.py
│   │   └── performance_optimizer.py
│   ├── visualization/            # Visualization tools
│   │   ├── building_visualizer.py
│   │   ├── ultra_advanced_visualizer.py
│   │   └── visualizer.py
│   ├── main_four_ap.py          # Main execution script
│   ├── advanced_heatmap_visualizer.py
│   ├── advanced_visualization.py
│   ├── enhanced_floor_plan_processor.py
│   └── floor_plan_analyzer.py
├── gui/                          # GUI applications
│   ├── run_gui.py               # Main GUI launcher
│   ├── wifi_gui_app.py          # Primary GUI application
│   ├── enhanced_3d_viewer.py    # 3D visualization
│   ├── simple_gui.py            # Simplified GUI
│   ├── working_gui.py           # Alternative GUI
│   └── requirements_gui.txt     # GUI-specific dependencies
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest configuration
│   ├── test_basic.py            # Basic functionality tests
│   └── test_propagation.py      # Propagation engine tests
├── floor_plans/                  # Building layout configurations
│   ├── floorplan.json
│   ├── floorplan.jpg
│   └── finalmap.json
├── docs/                         # Documentation
│   └── slides/                  # Presentation materials
├── runs/                         # Execution results
│   └── run_<timestamp>/         # Individual run outputs
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
├── README_SETUP.md              # Quick setup guide
├── SUMMARY.md                   # Project summary
├── TESTING.md                   # Testing guide
├── ARCHITECTURE.md              # This file
└── LICENSE                      # MIT License

```

## Core Components

### 1. Signal Propagation System

**Location**: `src/propagation/engines.py`

Implements multiple propagation models:
- **Fast Ray Tracing**: Optimized ray-based calculations
- **COST-231**: Industry-standard urban propagation model
- **VPLE**: Variable Path Loss Exponent model
- **Advanced Physics**: Full electromagnetic modeling

### 2. Material System

**Location**: `src/physics/materials.py`

- Frequency-dependent material properties
- Angle-dependent attenuation
- Composite material handling
- Temperature effects

### 3. AP Placement Optimization

**Location**: `src/main_four_ap.py`

- Genetic algorithm optimization
- Multi-objective optimization (coverage, cost, interference)
- Intelligent grid-based placement
- Material-aware positioning

### 4. Data Collection

**Location**: `src/data_collection/wifi_data_collector.py`

- RSSI calculation with multipath effects
- Free space path loss modeling
- Material attenuation integration
- Sample collection for ML training

### 5. Machine Learning Models

**Location**: `src/models/`

- Random Forest regression
- XGBoost/LightGBM support
- Gaussian Process Regression with uncertainty
- Hybrid physics-ML models
- Transfer learning utilities

### 6. Visualization

**Location**: `src/visualization/`

- Coverage heatmaps
- 3D signal mapping
- Building structure overlay
- Performance metrics visualization

## Data Flow

```
1. Configuration Input
   ↓
2. Building Layout Processing
   ↓
3. Material Grid Generation
   ↓
4. AP Placement Optimization
   ↓
5. Signal Propagation Calculation
   ↓
6. Coverage Analysis
   ↓
7. Visualization & Reporting
```

## Key Algorithms

### AP Placement Algorithm

1. **Estimation Phase**: Calculate initial AP count based on:
   - Building volume and area
   - User/device density
   - Material attenuation
   - Room structure

2. **Optimization Phase**: Use genetic algorithm to:
   - Maximize coverage
   - Minimize cost
   - Reduce interference
   - Optimize channel assignment

3. **Refinement Phase**: Apply constraints:
   - Minimum AP separation
   - Wall clearance
   - Ceiling mounting height
   - Room-based placement

### Signal Propagation

1. **Direct Path**: Free space path loss + material attenuation
2. **Reflected Paths**: Surface reflections with Fresnel coefficients
3. **Diffracted Paths**: Obstacle diffraction using Knife-edge model
4. **Multipath Combination**: Coherent power addition

## Configuration

### Floor Plan JSON Format

```json
{
  "building": {
    "width": 40.0,
    "length": 50.0,
    "height": 3.0
  },
  "regions": [
    {
      "name": "office",
      "type": "office",
      "material": "drywall",
      "coords": [x, y, width, height]
    }
  ],
  "aps": [
    {
      "x": 10.0,
      "y": 20.0,
      "z": 2.7,
      "tx_power": 20.0,
      "frequency": 2.4
    }
  ]
}
```

## Performance Considerations

### Optimization Strategies

1. **Quick Mode**: Reduced grid resolution and iterations
2. **Caching**: LRU cache for repeated calculations
3. **Parallel Processing**: Multi-threaded RSSI calculations
4. **Adaptive Resolution**: Variable grid density based on complexity

### Memory Management

- Lazy loading of materials grid
- Streaming data processing
- Efficient numpy array operations
- Garbage collection hints

## Extension Points

### Adding New Propagation Models

1. Inherit from `PropagationEngine` base class
2. Implement `calculate_rssi()` method
3. Register in engine factory

### Adding New Materials

1. Define material properties in `materials.py`
2. Add to `ADVANCED_MATERIALS` dictionary
3. Specify frequency-dependent properties

### Adding New Optimization Objectives

1. Define objective function in `main_four_ap.py`
2. Add to multi-objective optimization weights
3. Update fitness evaluation

## Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: Full pipeline validation
- **Performance Tests**: Benchmark critical paths
- **Regression Tests**: Prevent breaking changes

## Dependencies

### Core Dependencies
- numpy: Numerical computations
- scipy: Scientific algorithms
- scikit-learn: Machine learning
- matplotlib: Visualization
- pandas: Data manipulation

### Optional Dependencies
- xgboost: Advanced ML models
- lightgbm: Fast gradient boosting
- plotly: Interactive visualizations
- opencv: Image processing

## Future Enhancements

1. **Real-time Monitoring**: Live signal strength updates
2. **Multi-floor Support**: Enhanced 3D modeling
3. **Cloud Integration**: Distributed optimization
4. **Mobile App**: Field measurement integration
5. **AI-Powered Suggestions**: Automated troubleshooting
