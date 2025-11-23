# Testing Guide

## Test Suite Overview

The project includes comprehensive tests for core functionality:

- **Basic Tests** (`tests/test_basic.py`): WiFi data collection, materials, utilities, AP placement
- **Propagation Tests** (`tests/test_propagation.py`): Propagation engine implementations

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_basic.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_basic.py::TestWiFiDataCollector -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Pytest configuration and fixtures
├── test_basic.py        # Basic functionality tests
└── test_propagation.py  # Propagation engine tests
```

## Writing New Tests

### Example Test
```python
import pytest
from src.data_collection.wifi_data_collector import WiFiDataCollector

def test_my_feature():
    """Test description."""
    collector = WiFiDataCollector(tx_power=20.0)
    result = collector.calculate_rssi(10.0)
    assert result < 20.0
```

### Using Fixtures
```python
def test_with_fixture(wifi_collector):
    """Test using fixture from conftest.py."""
    rssi = wifi_collector.calculate_rssi(5.0)
    assert rssi > -100
```

## Current Test Coverage

- WiFi Data Collector: ✓ Fully tested
- Materials System: ✓ Basic tests
- Propagation Engines: ✓ All engines tested
- AP Placement: ✓ Core algorithms tested
- Utility Functions: ✓ Key functions tested

## Known Issues

- COST-231 engine has non-monotonic behavior at very short distances (< 5m)
- Some edge cases in material attenuation need additional tests

## Future Test Additions

- [ ] Integration tests for full pipeline
- [ ] Performance benchmarks
- [ ] GUI tests
- [ ] Visualization output validation
- [ ] Configuration file validation tests
