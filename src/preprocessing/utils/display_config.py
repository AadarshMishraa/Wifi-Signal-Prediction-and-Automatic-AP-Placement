"""
Configuration module for display settings and coordinate transformations.
Centralizes all dimension, scaling, and DPI settings to ensure consistency across visualizations.
"""

class DisplayConfig:
    # Output image settings
    DPI = 300
    OUTPUT_WIDTH = 3210   # Width at 300 DPI
    OUTPUT_HEIGHT = 1948  # Height at 300 DPI
    
    # Internal coordinate system (used by floor plan generator)
    INTERNAL_WIDTH = 1200
    INTERNAL_HEIGHT = 800
    
    # Scaling factors
    X_SCALE = OUTPUT_WIDTH / INTERNAL_WIDTH
    Y_SCALE = OUTPUT_HEIGHT / INTERNAL_HEIGHT
    
    # Standard figure sizes
    FIGURE_WIDTH = 12    # inches
    FIGURE_HEIGHT = 8    # inches
    
    # AP positioning constants (in output coordinates)
    AP_MARGIN_X = 600    # pixels from edge
    AP_MARGIN_Y = 365    # pixels from top/bottom
    
    @classmethod
    def to_output_coordinates(cls, x, y):
        """Convert internal coordinates to output coordinates."""
        return (x * cls.X_SCALE, y * cls.Y_SCALE)
    
    @classmethod
    def to_internal_coordinates(cls, x, y):
        """Convert output coordinates to internal coordinates."""
        return (x / cls.X_SCALE, y / cls.Y_SCALE)
    
    @classmethod
    def get_ap_positions(cls):
        """Get standard AP positions in output coordinates."""
        return [
            # Upper left
            (cls.AP_MARGIN_X, cls.AP_MARGIN_Y, "AP_UpperLeft"),
            # Upper right
            (cls.OUTPUT_WIDTH - cls.AP_MARGIN_X, cls.AP_MARGIN_Y, "AP_UpperRight"),
            # Lower left
            (cls.AP_MARGIN_X, cls.OUTPUT_HEIGHT - cls.AP_MARGIN_Y, "AP_LowerLeft"),
            # Lower right
            (cls.OUTPUT_WIDTH - cls.AP_MARGIN_X, cls.OUTPUT_HEIGHT - cls.AP_MARGIN_Y, "AP_LowerRight")
        ]
