"""
WiFiSignalPredictor: Unified ML/Physics/Hybrid WiFi Signal Prediction
- Supports advanced models, feature engineering, augmentation, uncertainty, transfer learning
"""
import numpy as np
from .wifi_models import WiFiModelFactory, fine_tune_model, HybridPhysicsMLModel
from src.preprocessing.data_augmentation import add_thermal_noise, add_interference, add_fading, simulate_environmental_variability
from src.preprocessing.feature_engineering import build_feature_matrix

class WiFiSignalPredictor:
    """
    Unified WiFi signal predictor supporting advanced ML, hybrid, and uncertainty-aware models.
    Usage:
        predictor = WiFiSignalPredictor(model_type='xgboost')
        predictor.fit(aps, rxs, obstacles, wall_segments, y)
        y_pred = predictor.predict(aps, rxs, obstacles, wall_segments)
    """
    def __init__(self, model_type='random_forest', model_kwargs=None, physics_model=None):
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.physics_model = physics_model
        self.model = WiFiModelFactory.create(model_type, **self.model_kwargs)
        self.is_fitted = False
    def fit(self, aps, rxs, obstacles, wall_segments, y, augment=True, fade_type='rayleigh'):
        """
        Fit the model. Optionally augment data with noise, interference, fading, and environmental variability.
        """
        X = build_feature_matrix(aps, rxs, obstacles, wall_segments)
        y_aug = y.copy()
        if augment:
            y_aug = add_thermal_noise(y_aug)
            y_aug = add_interference(y_aug)
            y_aug = add_fading(y_aug, fading_type=fade_type)
            # Optionally augment features
            # X = simulate_environmental_variability(X)  # Uncomment if using structured arrays
        self.model.fit(X, y_aug)
        self.is_fitted = True
        return self
    def predict(self, aps, rxs, obstacles, wall_segments, return_uncertainty=False):
        """
        Predict RSSI. If model supports uncertainty, return (mean, variance).
        """
        X = build_feature_matrix(aps, rxs, obstacles, wall_segments)
        if hasattr(self.model, 'predict'):
            if return_uncertainty and hasattr(self.model, 'predict') and 'return_std' in self.model.predict.__code__.co_varnames:
                mean, var = self.model.predict(X, return_std=True)
                return mean, var
            else:
                return self.model.predict(X)
        else:
            raise ValueError("Model does not support prediction")
    def fine_tune(self, aps, rxs, obstacles, wall_segments, y_new):
        """
        Fine-tune the model on new data (transfer learning).
        """
        X_new = build_feature_matrix(aps, rxs, obstacles, wall_segments)
        self.model = fine_tune_model(self.model, X_new, y_new)
        return self
    def set_physics_model(self, physics_model):
        """
        Set or update the physics model for hybrid use.
        """
        self.physics_model = physics_model
        if isinstance(self.model, HybridPhysicsMLModel):
            self.model.physics_model = physics_model