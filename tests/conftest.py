"""Pytest configuration and fixtures."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_building():
    """Fixture for a sample building configuration."""
    return {
        'width': 40.0,
        'length': 50.0,
        'height': 3.0,
        'resolution': 0.2
    }


@pytest.fixture
def sample_ap_locations():
    """Fixture for sample AP locations."""
    return {
        'AP1': (10.0, 10.0, 2.5),
        'AP2': (30.0, 10.0, 2.5),
        'AP3': (10.0, 40.0, 2.5),
        'AP4': (30.0, 40.0, 2.5)
    }


@pytest.fixture
def wifi_collector():
    """Fixture for WiFi data collector."""
    from src.data_collection.wifi_data_collector import WiFiDataCollector
    return WiFiDataCollector(tx_power=20.0, frequency=2.4e9)
