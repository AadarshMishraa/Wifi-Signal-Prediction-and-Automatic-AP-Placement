"""
Feature Engineering for WiFi ML
- Compute advanced features: distance to nearest obstacle, number of walls crossed, angle of incidence, etc.
"""
import numpy as np

def distance_to_nearest_obstacle(rx, obstacles):
    """Compute distance from receiver to nearest obstacle."""
    rx = np.array(rx)
    obstacles = np.array(obstacles)
    dists = np.linalg.norm(obstacles - rx, axis=1)
    return np.min(dists) if len(dists) > 0 else np.nan

def number_of_walls_crossed(ap, rx, wall_segments):
    """Estimate number of walls crossed between AP and receiver (stub)."""
    # For now, just count walls whose segment crosses the line (ap, rx)
    # wall_segments: list of ((x1, y1), (x2, y2))
    def crosses(ap, rx, wall):
        # Simple 2D line intersection stub
        return False  # TODO: Implement real geometry
    return sum(crosses(ap, rx, wall) for wall in wall_segments)

def angle_of_incidence(ap, rx, wall):
    """Compute angle of incidence at wall (stub)."""
    # wall: ((x1, y1), (x2, y2))
    # Return angle in degrees
    return 0.0  # TODO: Implement real geometry

def build_feature_matrix(aps, rxs, obstacles, wall_segments):
    """Build feature matrix for ML model."""
    features = []
    for ap, rx in zip(aps, rxs):
        d_nearest = distance_to_nearest_obstacle(rx, obstacles)
        n_walls = number_of_walls_crossed(ap, rx, wall_segments)
        angle = 0.0  # Could loop over walls for real angle
        features.append([d_nearest, n_walls, angle])
    return np.array(features) 