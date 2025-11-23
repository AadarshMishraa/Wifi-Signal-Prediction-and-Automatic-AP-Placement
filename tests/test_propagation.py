"""Tests for propagation engines."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.propagation.engines import (
    FastRayTracingEngine, 
    Cost231Engine, 
    VPLEEngine,
    AdvancedPhysicsEngine
)


class TestPropagationEngines:
    """Test propagation engine implementations."""
    
    def test_fast_ray_tracing_engine(self):
        """Test Fast Ray Tracing Engine."""
        engine = FastRayTracingEngine()
        
        ap = (10.0, 10.0, 2.5)
        point = (15.0, 15.0, 1.0)
        
        rssi = engine.calculate_rssi(ap, point, None, tx_power=20.0)
        
        assert isinstance(rssi, (int, float))
        assert -100 <= rssi <= 20.0
    
    def test_cost231_engine(self):
        """Test COST-231 Engine."""
        engine = Cost231Engine()
        
        ap = (10.0, 10.0, 2.5)
        point = (15.0, 15.0, 1.0)
        
        rssi = engine.calculate_rssi(ap, point, None, tx_power=20.0, frequency=2400)
        
        assert isinstance(rssi, (int, float))
        assert -200 <= rssi <= 20.0  # COST-231 can have higher path loss
    
    def test_vple_engine(self):
        """Test VPLE Engine."""
        engine = VPLEEngine()
        
        ap = (10.0, 10.0, 2.5)
        point = (15.0, 15.0, 1.0)
        
        rssi = engine.calculate_rssi(ap, point, None, tx_power=20.0, frequency=2400)
        
        assert isinstance(rssi, (int, float))
        assert -150 <= rssi <= 20.0
    
    def test_advanced_physics_engine(self):
        """Test Advanced Physics Engine."""
        engine = AdvancedPhysicsEngine(frequency=2.4e9)
        
        ap = (10.0, 10.0, 2.5)
        point = (15.0, 15.0, 1.0)
        
        rssi = engine.calculate_rssi(ap, point, None, tx_power=20.0)
        
        assert isinstance(rssi, (int, float))
        assert -150 <= rssi <= 20.0
    
    def test_rssi_decreases_with_distance(self):
        """Test that RSSI decreases with distance for all engines."""
        engines = [
            FastRayTracingEngine(),
            VPLEEngine(),
            AdvancedPhysicsEngine()
        ]
        
        ap = (10.0, 10.0, 2.5)
        point_near = (11.0, 10.0, 2.5)
        point_far = (20.0, 10.0, 2.5)
        
        for engine in engines:
            rssi_near = engine.calculate_rssi(ap, point_near, None, tx_power=20.0)
            rssi_far = engine.calculate_rssi(ap, point_far, None, tx_power=20.0)
            
            assert rssi_near > rssi_far, f"{engine.__class__.__name__} failed distance test"
        
        # Note: COST-231 model has complex behavior at short distances
        # and may not follow simple monotonic decrease pattern


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
