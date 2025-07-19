"""
Comprehensive Error Handling and Logging System

This module provides:
- Robust exception handling for all critical operations
- Comprehensive input validation
- Detailed logging at multiple levels
- Performance monitoring and profiling
- Graceful degradation and fallback mechanisms
"""

import logging
import traceback
import sys
import time
import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import json
import inspect

class LogLevel(Enum):
    """Log levels for different types of information."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationError:
    """Structured validation error information."""
    field_name: str
    value: Any
    expected_type: str
    constraint: str
    severity: ErrorSeverity
    message: str

@dataclass
class PerformanceMetric:
    """Performance metric tracking."""
    operation_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

class ErrorHandler:
    """
    Comprehensive error handling and logging system.
    """
    
    def __init__(self, log_file: Optional[str] = None, log_level: LogLevel = LogLevel.INFO):
        """Initialize the error handler."""
        self.log_file = log_file
        self.log_level = log_level
        self.validation_errors: List[ValidationError] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.error_count = 0
        self.warning_count = 0
        
        # Setup logging
        self._setup_logging()
        
        # Performance tracking
        self.operation_timers = {}
        
        logger.info("Error Handler initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.value))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level.value))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Suppress warnings from specific libraries
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
        
        logger.info(f"Logging configured - Level: {self.log_level.value}, File: {self.log_file}")
    
    def validate_input(self, value: Any, expected_type: Union[type, Tuple[type, ...]], 
                      field_name: str = "", constraints: Dict[str, Any] = None) -> bool:
        """
        Validate input with comprehensive error reporting.
        
        Args:
            value: Value to validate
            expected_type: Expected type(s)
            field_name: Name of the field being validated
            constraints: Additional constraints (min, max, pattern, etc.)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Type validation
            if not isinstance(value, expected_type):
                error = ValidationError(
                    field_name=field_name,
                    value=value,
                    expected_type=str(expected_type),
                    constraint="type",
                    severity=ErrorSeverity.HIGH,
                    message=f"Expected {expected_type}, got {type(value)}"
                )
                self.validation_errors.append(error)
                logger.error(f"Validation error: {error.message}")
                return False
            
            # Additional constraints
            if constraints:
                if not self._check_constraints(value, constraints, field_name):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during validation of {field_name}: {e}")
            return False
    
    def _check_constraints(self, value: Any, constraints: Dict[str, Any], field_name: str) -> bool:
        """Check additional constraints on a value."""
        try:
            # Numeric constraints
            if isinstance(value, (int, float, np.number)):
                if 'min' in constraints and value < constraints['min']:
                    error = ValidationError(
                        field_name=field_name,
                        value=value,
                        expected_type="numeric",
                        constraint=f"min={constraints['min']}",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Value {value} is below minimum {constraints['min']}"
                    )
                    self.validation_errors.append(error)
                    logger.warning(f"Constraint violation: {error.message}")
                    return False
                
                if 'max' in constraints and value > constraints['max']:
                    error = ValidationError(
                        field_name=field_name,
                        value=value,
                        expected_type="numeric",
                        constraint=f"max={constraints['max']}",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Value {value} is above maximum {constraints['max']}"
                    )
                    self.validation_errors.append(error)
                    logger.warning(f"Constraint violation: {error.message}")
                    return False
            
            # String constraints
            if isinstance(value, str):
                if 'min_length' in constraints and len(value) < constraints['min_length']:
                    error = ValidationError(
                        field_name=field_name,
                        value=value,
                        expected_type="string",
                        constraint=f"min_length={constraints['min_length']}",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"String length {len(value)} is below minimum {constraints['min_length']}"
                    )
                    self.validation_errors.append(error)
                    logger.warning(f"Constraint violation: {error.message}")
                    return False
                
                if 'max_length' in constraints and len(value) > constraints['max_length']:
                    error = ValidationError(
                        field_name=field_name,
                        value=value,
                        expected_type="string",
                        constraint=f"max_length={constraints['max_length']}",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"String length {len(value)} is above maximum {constraints['max_length']}"
                    )
                    self.validation_errors.append(error)
                    logger.warning(f"Constraint violation: {error.message}")
                    return False
            
            # Array/list constraints
            if isinstance(value, (list, np.ndarray)):
                if 'min_length' in constraints and len(value) < constraints['min_length']:
                    error = ValidationError(
                        field_name=field_name,
                        value=value,
                        expected_type="array",
                        constraint=f"min_length={constraints['min_length']}",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Array length {len(value)} is below minimum {constraints['min_length']}"
                    )
                    self.validation_errors.append(error)
                    logger.warning(f"Constraint violation: {error.message}")
                    return False
                
                if 'max_length' in constraints and len(value) > constraints['max_length']:
                    error = ValidationError(
                        field_name=field_name,
                        value=value,
                        expected_type="array",
                        constraint=f"max_length={constraints['max_length']}",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Array length {len(value)} is above maximum {constraints['max_length']}"
                    )
                    self.validation_errors.append(error)
                    logger.warning(f"Constraint violation: {error.message}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking constraints for {field_name}: {e}")
            return False
    
    def safe_operation(self, operation: Callable, *args, fallback_value: Any = None, 
                      operation_name: str = "", **kwargs) -> Any:
        """
        Execute an operation with comprehensive error handling.
        
        Args:
            operation: Function to execute
            *args: Arguments for the operation
            fallback_value: Value to return if operation fails
            operation_name: Name of the operation for logging
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of operation or fallback value
        """
        if not operation_name:
            operation_name = operation.__name__
        
        start_time = time.time()
        
        try:
            logger.debug(f"Starting operation: {operation_name}")
            
            # Execute operation
            result = operation(*args, **kwargs)
            
            # Record performance
            execution_time = time.time() - start_time
            metric = PerformanceMetric(
                operation_name=operation_name,
                execution_time=execution_time
            )
            self.performance_metrics.append(metric)
            
            logger.debug(f"Operation {operation_name} completed in {execution_time:.4f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.error_count += 1
            
            # Log detailed error information
            logger.error(f"Operation {operation_name} failed after {execution_time:.4f}s")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Record failed operation
            metric = PerformanceMetric(
                operation_name=f"{operation_name}_FAILED",
                execution_time=execution_time
            )
            self.performance_metrics.append(metric)
            
            if fallback_value is not None:
                logger.info(f"Using fallback value for {operation_name}")
                return fallback_value
            else:
                raise
    
    def performance_monitor(self, operation_name: str = ""):
        """
        Decorator for performance monitoring.
        
        Usage:
            @error_handler.performance_monitor("my_operation")
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or func.__name__
                return self.safe_operation(func, *args, operation_name=name, **kwargs)
            return wrapper
        return decorator
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary with comprehensive checks.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating configuration...")
        
        # Required fields
        required_fields = {
            'building_width': (float, {'min': 0.1, 'max': 1000.0}),
            'building_length': (float, {'min': 0.1, 'max': 1000.0}),
            'building_height': (float, {'min': 0.1, 'max': 100.0}),
            'target_coverage': (float, {'min': 0.0, 'max': 1.0}),
        }
        
        for field_name, (expected_type, constraints) in required_fields.items():
            if field_name not in config:
                error = ValidationError(
                    field_name=field_name,
                    value=None,
                    expected_type=str(expected_type),
                    constraint="required",
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Required field '{field_name}' is missing"
                )
                self.validation_errors.append(error)
                logger.error(f"Missing required field: {field_name}")
                return False
            
            if not self.validate_input(config[field_name], expected_type, field_name, constraints):
                return False
        
        # Optional fields with validation
        optional_fields = {
            'tx_power': (float, {'min': -10.0, 'max': 30.0}),
            'frequency': (float, {'min': 1e9, 'max': 10e9}),
            'noise_floor': (float, {'min': -120.0, 'max': -50.0}),
        }
        
        for field_name, (expected_type, constraints) in optional_fields.items():
            if field_name in config:
                if not self.validate_input(config[field_name], expected_type, field_name, constraints):
                    logger.warning(f"Optional field {field_name} has invalid value")
        
        logger.info("Configuration validation completed")
        return len([e for e in self.validation_errors if e.severity == ErrorSeverity.CRITICAL]) == 0
    
    def validate_materials_grid(self, materials_grid: np.ndarray, 
                              expected_shape: Tuple[int, int, int]) -> bool:
        """
        Validate materials grid with comprehensive checks.
        
        Args:
            materials_grid: 3D materials grid to validate
            expected_shape: Expected shape (z, y, x)
            
        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating materials grid...")
        
        # Type validation
        if not self.validate_input(materials_grid, np.ndarray, "materials_grid"):
            return False
        
        # Shape validation
        if materials_grid.shape != expected_shape:
            error = ValidationError(
                field_name="materials_grid",
                value=materials_grid.shape,
                expected_type=f"shape {expected_shape}",
                constraint="shape",
                severity=ErrorSeverity.CRITICAL,
                message=f"Expected shape {expected_shape}, got {materials_grid.shape}"
            )
            self.validation_errors.append(error)
            logger.error(f"Materials grid shape mismatch: {error.message}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(materials_grid)):
            logger.warning("Materials grid contains NaN values")
        
        if np.any(np.isinf(materials_grid)):
            logger.warning("Materials grid contains infinite values")
        
        # Check for negative material IDs
        if np.any(materials_grid < 0):
            logger.warning("Materials grid contains negative material IDs")
        
        logger.info("Materials grid validation completed")
        return True
    
    def validate_ap_locations(self, ap_locations: List[Tuple[float, float, float]], 
                            building_dimensions: Tuple[float, float, float]) -> bool:
        """
        Validate AP locations with boundary checks.
        
        Args:
            ap_locations: List of AP coordinates
            building_dimensions: Building dimensions (width, length, height)
            
        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating AP locations...")
        
        if not self.validate_input(ap_locations, list, "ap_locations"):
            return False
        
        width, length, height = building_dimensions
        
        for i, ap_location in enumerate(ap_locations):
            if not self.validate_input(ap_location, tuple, f"ap_location_{i}"):
                return False
            
            if len(ap_location) != 3:
                error = ValidationError(
                    field_name=f"ap_location_{i}",
                    value=ap_location,
                    expected_type="tuple of length 3",
                    constraint="length",
                    severity=ErrorSeverity.HIGH,
                    message=f"AP location must have 3 coordinates, got {len(ap_location)}"
                )
                self.validation_errors.append(error)
                logger.error(f"AP location validation error: {error.message}")
                return False
            
            x, y, z = ap_location
            
            # Check bounds
            if not (0 <= x <= width):
                logger.warning(f"AP {i} x-coordinate {x} is outside building width [0, {width}]")
            
            if not (0 <= y <= length):
                logger.warning(f"AP {i} y-coordinate {y} is outside building length [0, {length}]")
            
            if not (0 <= z <= height):
                logger.warning(f"AP {i} z-coordinate {z} is outside building height [0, {height}]")
        
        logger.info("AP locations validation completed")
        return True
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        critical_errors = [e for e in self.validation_errors if e.severity == ErrorSeverity.CRITICAL]
        high_errors = [e for e in self.validation_errors if e.severity == ErrorSeverity.HIGH]
        medium_errors = [e for e in self.validation_errors if e.severity == ErrorSeverity.MEDIUM]
        low_errors = [e for e in self.validation_errors if e.severity == ErrorSeverity.LOW]
        
        return {
            'total_errors': len(self.validation_errors),
            'critical_errors': len(critical_errors),
            'high_errors': len(high_errors),
            'medium_errors': len(medium_errors),
            'low_errors': len(low_errors),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'validation_passed': len(critical_errors) == 0,
            'performance_metrics': {
                'total_operations': len(self.performance_metrics),
                'avg_execution_time': np.mean([m.execution_time for m in self.performance_metrics]) if self.performance_metrics else 0.0,
                'max_execution_time': max([m.execution_time for m in self.performance_metrics]) if self.performance_metrics else 0.0,
                'min_execution_time': min([m.execution_time for m in self.performance_metrics]) if self.performance_metrics else 0.0,
            },
            'detailed_errors': [
                {
                    'field_name': e.field_name,
                    'severity': e.severity.value,
                    'message': e.message
                }
                for e in self.validation_errors
            ]
        }
    
    def save_error_report(self, filepath: str):
        """Save error report to file."""
        try:
            report = self.get_validation_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Error report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving error report: {e}")
    
    def clear_errors(self):
        """Clear all error tracking."""
        self.validation_errors.clear()
        self.performance_metrics.clear()
        self.error_count = 0
        self.warning_count = 0
        logger.info("Error tracking cleared")

# Global error handler instance
error_handler = ErrorHandler()

def test_error_handling():
    """Test the error handling system."""
    print("Testing Error Handling System...")
    
    # Test input validation
    assert error_handler.validate_input(5, int, "test_int", {'min': 0, 'max': 10})
    assert not error_handler.validate_input(-1, int, "test_int", {'min': 0, 'max': 10})
    
    # Test safe operation
    def test_function(x, y):
        return x + y
    
    result = error_handler.safe_operation(test_function, 2, 3, operation_name="test_add")
    assert result == 5
    
    # Test operation with error
    def failing_function():
        raise ValueError("Test error")
    
    result = error_handler.safe_operation(failing_function, fallback_value=42)
    assert result == 42
    
    # Test performance monitoring decorator
    @error_handler.performance_monitor("decorated_function")
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    result = slow_function()
    assert result == "done"
    
    # Test configuration validation
    config = {
        'building_width': 50.0,
        'building_length': 30.0,
        'building_height': 3.0,
        'target_coverage': 0.9,
        'tx_power': 20.0
    }
    
    assert error_handler.validate_config(config)
    
    # Test materials grid validation
    materials_grid = np.random.randint(0, 5, (10, 20, 30))
    assert error_handler.validate_materials_grid(materials_grid, (10, 20, 30))
    
    # Test AP locations validation
    ap_locations = [(10.0, 15.0, 2.7), (25.0, 10.0, 2.7)]
    building_dimensions = (50.0, 30.0, 3.0)
    assert error_handler.validate_ap_locations(ap_locations, building_dimensions)
    
    # Get validation report
    report = error_handler.get_validation_report()
    print(f"Validation report: {report}")
    
    print("Error Handling System test completed successfully!")

if __name__ == "__main__":
    test_error_handling() 