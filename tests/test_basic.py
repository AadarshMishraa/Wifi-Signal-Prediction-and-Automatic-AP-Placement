"""Basic tests for core functionality."""

import sys
import os
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.wifi_data_collector import WiFiDataCollector
from src.physics.materials import MATERIALS, ADVANCED_MATERIALS


class TestWiFiDataCollector:
    """Test WiFi data collection functionality."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = WiFiDataCollector(tx_power=20.0, frequency=2.4e9)
        assert collector.tx_power == 20.0
        assert collector.frequency == 2.4e9
        assert collector.noise_floor == -96.0
    
    def test_free_space_loss(self):
        """Test free space path loss calculation."""
        collector = WiFiDataCollector(tx_power=20.0, frequency=2.4e9)
        
        # At 1 meter, should have some loss
        loss_1m = collector.calculate_free_space_loss(1.0)
        assert loss_1m > 0
        
        # At 10 meters, should have more loss
        loss_10m = collector.calculate_free_space_loss(10.0)
        assert loss_10m > loss_1m
    
    def test_rssi_calculation(self):
        """Test RSSI calculation."""
        collector = WiFiDataCollector(tx_power=20.0, frequency=2.4e9)
        
        # RSSI should decrease with distance
        rssi_1m = collector.calculate_rssi(1.0, include_multipath=False)
        rssi_10m = collector.calculate_rssi(10.0, include_multipath=False)
        
        assert rssi_1m > rssi_10m
        assert rssi_1m <= 20.0  # Should not exceed tx_power
        assert rssi_10m >= -96.0  # Should not go below noise floor
    
    def test_collect_samples(self):
        """Test sample collection."""
        collector = WiFiDataCollector(tx_power=20.0, frequency=2.4e9)
        
        points = [(0, 0), (1, 0), (0, 1), (1, 1)]
        ap_location = (0.5, 0.5)
        
        samples = collector.collect_samples(points, ap_location)
        
        assert len(samples) == len(points)
        assert all(isinstance(s, (int, float, np.number)) for s in samples)
        assert all(s >= -96.0 for s in samples)  # Above noise floor


class TestMaterials:
    """Test material definitions."""
    
    def test_materials_exist(self):
        """Test that material databases are populated."""
        assert len(MATERIALS) > 0 or len(ADVANCED_MATERIALS) > 0
    
    def test_air_material(self):
        """Test that air material exists."""
        assert 'air' in ADVANCED_MATERIALS
        air = ADVANCED_MATERIALS['air']
        assert air.name.lower() == 'air'


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_distance_calculation(self):
        """Test 3D distance calculation."""
        from src.main_four_ap import distance_3d
        
        # Test basic distance
        d = distance_3d((0, 0, 0), (3, 4, 0))
        assert abs(d - 5.0) < 0.01
        
        # Test 3D distance
        d = distance_3d((0, 0, 0), (1, 1, 1))
        assert abs(d - np.sqrt(3)) < 0.01
    
    def test_ap_list_to_dict(self):
        """Test AP list to dictionary conversion."""
        from src.main_four_ap import ap_list_to_dict
        
        # Test with flat list
        ap_list = [10.0, 20.0, 2.5, 15.0, 30.0, 40.0, 2.5, 15.0]
        ap_dict = ap_list_to_dict(ap_list)
        
        assert len(ap_dict) == 2
        assert 'AP1' in ap_dict
        assert 'AP2' in ap_dict
        assert ap_dict['AP1'] == (10.0, 20.0, 2.5, 15.0)


class TestAPPlacement:
    """Test AP placement algorithms."""
    
    def test_intelligent_grid_placement(self):
        """Test intelligent grid-based AP placement."""
        from src.main_four_ap import _place_aps_intelligent_grid
        
        # Test basic placement
        ap_locations = _place_aps_intelligent_grid(
            num_aps=4,
            building_width=20.0,
            building_length=30.0,
            building_height=3.0,
            materials_grid=None
        )
        
        assert len(ap_locations) == 4
        assert all(isinstance(k, str) for k in ap_locations.keys())
        assert all(len(v) == 3 for v in ap_locations.values())
        
        # Check that APs are within building bounds
        for ap_name, (x, y, z) in ap_locations.items():
            assert 0 <= x <= 20.0
            assert 0 <= y <= 30.0
            assert 0 <= z <= 3.0
    
    def test_ap_estimation(self):
        """Test AP count estimation."""
        from src.main_four_ap import estimate_initial_ap_count
        
        # Small building should need fewer APs
        count_small, _ = estimate_initial_ap_count(10.0, 10.0, 3.0)
        
        # Large building should need more APs
        count_large, _ = estimate_initial_ap_count(50.0, 50.0, 3.0)
        
        assert count_small > 0
        assert count_large > count_small


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
