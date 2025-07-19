# WiFi Signal Prediction System Architecture

## System Components

### 1. Data Collection Module
- WiFi Data Collector: Simulates signal strength measurements
- Material Physics Engine: Models signal attenuation through different materials
- Sampling Grid: High-resolution 200x120 point sampling

### 2. Physics Simulation
- Material Properties:
  - Concrete, Glass, Wood, Drywall
  - Each with specific permittivity and conductivity values
  - Thickness-based attenuation modeling

### 3. Visualization System
- Building Layout Engine
  - Material Grid System (0.1m resolution)
  - Complex Office Layout Support
  - Multi-layer Material Handling

- Signal Visualization
  - Heatmap Generation
  - Gaussian Interpolation
  - Material Overlay System
  - Access Point Markers

### 4. Data Flow
1. Building Layout Definition → Material Grid
2. AP Placement → Signal Source Points
3. Physics-based Signal Propagation
4. Data Collection & Processing
5. Visualization Generation
