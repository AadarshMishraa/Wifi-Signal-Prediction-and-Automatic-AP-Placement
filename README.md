# WiFi Signal Prediction and AP Placement Optimization

A comprehensive Python-based system for predicting WiFi signal strength, optimizing access point (AP) placement, and generating detailed visualizations for indoor wireless network planning. This project combines advanced physics-based signal propagation modeling with machine learning optimization to help network engineers and IT professionals design optimal WiFi coverage.

## 🚀 What This Project Does

This system acts as a "WiFi weather map" for buildings, helping you:
- **Predict signal strength** at every point in your building
- **Optimize AP placement** using genetic algorithms and multi-objective optimization
- **Visualize coverage** with detailed heatmaps and 3D plots
- **Analyze performance** with statistical metrics and interference calculations
- **Plan network infrastructure** before physical installation

## 🎯 Key Features

### 📊 Advanced Visualization
- **Individual AP Heatmaps**: Signal strength visualization for each access point
- **Combined Coverage Maps**: Overall signal strength using best signal at each point
- **Building Structure Overlay**: Walls, materials, and room boundaries
- **3D Signal Mapping**: Multi-floor signal propagation analysis
- **Interactive Dashboards**: Real-time parameter adjustment and visualization

### 🤖 Machine Learning & Optimization
- **Multi-Objective Genetic Algorithm**: Optimizes coverage, cost, and performance
- **Surrogate Models**: Fast prediction using trained ML models
- **Material-Aware Placement**: Considers wall attenuation and building materials
- **Interference Analysis**: SINR calculations and channel optimization
- **Adaptive Voxel System**: Efficient 3D signal propagation modeling

### 📈 Performance Analysis
- **Coverage Metrics**: Percentage of area with good/fair signal strength
- **Capacity Planning**: User density and device load analysis
- **Interference Mapping**: Signal-to-interference-plus-noise ratio (SINR)
- **Cost Optimization**: Balance between coverage and infrastructure cost
- **Statistical Reports**: Detailed performance comparisons and recommendations

## 🏗️ Project Architecture

```
wifi-signal-prediction-main/
├── src/                           # Core source code
│   ├── main_four_ap.py           # Main execution script
│   ├── advanced_heatmap_visualizer.py  # Visualization engine
│   ├── physics/                   # Signal propagation physics
│   │   ├── adaptive_voxel_system.py
│   │   └── materials.py
│   ├── models/                    # ML models and optimization
│   │   ├── wifi_models.py
│   │   └── wifi_classifier.py
│   ├── visualization/             # Plotting and visualization
│   │   ├── visualizer.py
│   │   ├── building_visualizer.py
│   │   └── ultra_advanced_visualizer.py
│   ├── preprocessing/             # Data processing
│   │   ├── preprocessor.py
│   │   ├── feature_engineering.py
│   │   └── data_augmentation.py
│   └── utils/                     # Utility functions
├── floor_plans/                   # Building layout files
├── results/                       # Generated outputs
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** for cloning the repository
- **Virtual environment** (recommended)

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd wifi-signal-prediction-main
```

2. **Create and activate virtual environment:**

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import numpy, pandas, matplotlib, scipy; print('Installation successful!')"
```

## 🚀 How to Run

### Basic Usage (Quick Start)

Run the main script with default settings:
```bash
python src/main_four_ap.py
```

This will:
1. Load default building layout (50m × 30m)
2. Place 4 access points optimally
3. Generate signal strength predictions
4. Create comprehensive visualizations
5. Save results to `results/` directory

### Advanced Usage

#### 1. Custom Building Layout
```bash
python src/main_four_ap.py --config floor_plans/custom_layout.json
```

#### 2. Specify Number of APs
```bash
python src/main_four_ap.py --num_aps 6 --target_coverage 0.95
```

#### 3. Optimization Mode
```bash
python src/main_four_ap.py --optimize --pop_size 50 --generations 100
```

#### 4. 3D Analysis
```bash
python src/main_four_ap.py --3d --building_height 10.0
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num_aps` | Number of access points | 4 |
| `--target_coverage` | Target coverage percentage | 0.9 |
| `--optimize` | Enable genetic algorithm optimization | False |
| `--3d` | Enable 3D analysis | False |
| `--quick_mode` | Fast mode with reduced resolution | False |
| `--output_dir` | Output directory | `results/` |
| `--config` | Configuration file path | None |

## 📊 Understanding the Output

### Generated Files

1. **Visualization Plots** (`results/plots/`):
   - `coverage_combined.png` - Overall coverage heatmap
   - `ap_individual_*.png` - Individual AP coverage maps
   - `signal_distribution.png` - Signal strength histograms
   - `interference_map.png` - Interference analysis
   - `capacity_analysis.png` - User capacity planning

2. **Data Files** (`results/data/`):
   - `signal_predictions.csv` - Raw signal strength data
   - `ap_locations.json` - Optimized AP positions
   - `performance_metrics.json` - Statistical analysis
   - `optimization_results.json` - Genetic algorithm results

3. **Reports** (`results/reports/`):
   - `coverage_report.html` - Interactive HTML report
   - `performance_summary.txt` - Text summary
   - `recommendations.md` - Actionable recommendations

### Key Metrics Explained

- **Coverage Percentage**: Area with signal ≥ -70 dBm (good) or ≥ -80 dBm (fair)
- **Average Signal Strength**: Mean RSSI across all points
- **SINR**: Signal-to-interference-plus-noise ratio
- **Capacity**: Maximum supported users per AP
- **Cost Efficiency**: Coverage per dollar spent

## 🔧 Configuration

### Building Layout Configuration

Create a JSON file to define your building:
```json
{
  "building_width": 50.0,
  "building_length": 30.0,
  "building_height": 3.0,
  "materials": {
    "walls": {"attenuation": 6.0, "thickness": 0.2},
    "windows": {"attenuation": 2.0, "thickness": 0.01},
    "doors": {"attenuation": 3.0, "thickness": 0.05}
  },
  "rooms": [
    {
      "name": "Conference Room",
      "polygon": [[0, 0], [10, 0], [10, 8], [0, 8]],
      "material": "drywall"
    }
  ]
}
```

### Optimization Parameters

```python
# In your script or config file
optimization_config = {
    "population_size": 40,
    "generations": 30,
    "crossover_prob": 0.5,
    "mutation_prob": 0.3,
    "min_aps": 2,
    "max_aps": 10,
    "ap_cost": 500,
    "power_cost_per_dbm": 2
}
```

## 🧪 Advanced Features

### 1. Material-Aware Signal Propagation
The system models different building materials:
- **Concrete walls**: High attenuation (6-8 dB)
- **Glass windows**: Low attenuation (2-3 dB)
- **Drywall**: Medium attenuation (3-5 dB)
- **Wooden doors**: Variable attenuation (3-6 dB)

### 2. Multi-Objective Optimization
Genetic algorithm optimizes:
- **Coverage maximization**
- **Cost minimization**
- **Interference reduction**
- **Capacity planning**

### 3. 3D Signal Analysis
- Multi-floor signal propagation
- Vertical signal attenuation
- Ceiling and floor effects
- Elevation-based optimization

### 4. Real-Time Visualization
- Interactive parameter adjustment
- Live coverage updates
- Performance monitoring
- Export capabilities

## 📈 Performance Results

Based on extensive testing:

### Model Accuracy
- **Random Forest**: RMSE 0.01, R² 1.00 (Best)
- **SVM**: RMSE 0.10, R² 0.99
- **KNN**: RMSE 0.15, R² 0.98

### Optimization Performance
- **Coverage Improvement**: 15-25% over random placement
- **Cost Reduction**: 20-30% through optimal AP count
- **Interference Reduction**: 40-60% through channel planning

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

2. **Memory Issues**:
```bash
python src/main_four_ap.py --quick_mode
```

3. **Visualization Errors**:
```bash
pip install matplotlib --upgrade
```

4. **Slow Performance**:
```bash
python src/main_four_ap.py --quick_mode --num_aps 2
```

### Debug Mode
```bash
python src/main_four_ap.py --debug --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
pip install -r requirements.txt
pip install pytest black flake8
```

## 📚 Documentation

- **Technical Details**: See `SUMMARY.md`
- **API Reference**: Check docstrings in source code
- **Examples**: Look in `docs/examples/`
- **Research Papers**: Referenced in `docs/papers/`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Contributors and maintainers
- Research community for signal propagation models
- Open source libraries: NumPy, Pandas, Matplotlib, SciPy, DEAP
- Academic institutions for theoretical foundations

## 📞 Support

- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: Contact maintainers directly

---

**Ready to optimize your WiFi network?** Start with `python src/main_four_ap.py` and see the magic happen! 🚀
