# Quick Setup Guide

## Prerequisites
- Python 3.8+ (tested with Python 3.13)
- macOS, Linux, or Windows

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Wifi-Signal-Prediction-and-Automatic-AP-Placement
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Development Dependencies (Optional)
```bash
pip install pytest pytest-cov
```

## Quick Start

### Run with Default Settings
```bash
python src/main_four_ap.py --quick-mode
```

### Run with Custom Building
```bash
python src/main_four_ap.py --width 30 --length 40 --height 3 --quick-mode
```

### Run with Floor Plan Configuration
```bash
python src/main_four_ap.py --floor-plan-config floor_plans/floorplan.json
```

## Running Tests

```bash
pytest tests/ -v
```

## GUI Applications

The project includes multiple GUI implementations:

### Main GUI (Recommended)
```bash
python gui/run_gui.py
```

### Simple GUI
```bash
python gui/simple_gui.py
```

### Working GUI
```bash
python gui/working_gui.py
```

## Output

Results are saved to `runs/run_<timestamp>/`:
- `plots/` - Visualization images
- `data/` - Raw data files
- `run_info.json` - Configuration and metadata
- `performance_report.json` - Performance metrics

## Troubleshooting

### Virtual Environment Issues
If you encounter venv issues:
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Import Errors
Make sure you're in the project root directory and the virtual environment is activated.

### Memory Issues
Use `--quick-mode` flag to reduce memory usage:
```bash
python src/main_four_ap.py --quick-mode
```

## Documentation

- Full documentation: See `README.md`
- Project summary: See `SUMMARY.md`
- Presentation slides: See `docs/slides/`

## Support

For issues and questions, please create a GitHub issue.
