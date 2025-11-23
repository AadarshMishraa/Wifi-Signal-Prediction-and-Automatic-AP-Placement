#!/usr/bin/env python3
"""
Main launcher script for WiFi AP Placement GUI
Integrates the enhanced 3D viewer with the main GUI application
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'src'))
sys.path.append(current_dir)

try:
    # Import step by step to identify issues
    print("Loading GUI components...")
    
    # Set matplotlib backend first
    import matplotlib
    matplotlib.use('TkAgg')
    
    from wifi_gui_app import WiFiGUIApp, FloorPlanCanvas
    print("✓ Main GUI loaded")
    
    from enhanced_3d_viewer import Enhanced3DViewer
    print("✓ 3D Viewer loaded")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to simple GUI...")
    
    # Try simple GUI as fallback
    try:
        from simple_gui import SimpleWiFiGUI as WiFiGUIApp
        print("✓ Simple GUI loaded as fallback")
        SIMPLE_MODE = True
    except ImportError:
        print("Please ensure all required packages are installed:")
        print("pip install -r requirements_gui.txt")
        sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)

class IntegratedWiFiGUI:
    """Integrated WiFi GUI with enhanced 3D visualization"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WiFi AP Placement Designer - Enhanced Edition")
        self.root.geometry("1600x1000")
        
        # Set up the main interface
        self.setup_integrated_ui()
        
    def setup_integrated_ui(self):
        """Setup the integrated user interface"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for floor plan drawing (40% width)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Right panel for enhanced visualization (60% width)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        self.setup_floor_plan_panel(left_frame)
        self.setup_visualization_panel(right_frame)
        
    def setup_floor_plan_panel(self, parent):
        """Setup the floor plan drawing panel"""
        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Drawing mode buttons
        ttk.Label(toolbar, text="Drawing Mode:").pack(side=tk.LEFT, padx=5)
        
        self.drawing_mode = tk.StringVar(value='wall')
        modes = [('Wall', 'wall'), ('Room', 'room'), ('Obstacle', 'obstacle'), ('AP', 'ap')]
        
        for text, mode in modes:
            ttk.Radiobutton(toolbar, text=text, variable=self.drawing_mode, 
                           value=mode, command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        ttk.Button(toolbar, text="Clear All", command=self.clear_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="Auto Place APs", command=self.auto_place_aps).pack(side=tk.RIGHT, padx=2)
        
        # Building parameters frame
        params_frame = ttk.LabelFrame(parent, text="Building Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters in a grid
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_grid, text="Height (m):").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.height_var = tk.DoubleVar(value=3.0)
        ttk.Entry(params_grid, textvariable=self.height_var, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(params_grid, text="Coverage (%):").grid(row=0, column=2, sticky=tk.W, padx=2)
        self.coverage_var = tk.DoubleVar(value=90.0)
        ttk.Entry(params_grid, textvariable=self.coverage_var, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(params_grid, text="Resolution:").grid(row=1, column=0, sticky=tk.W, padx=2)
        self.resolution_var = tk.IntVar(value=100)
        ttk.Entry(params_grid, textvariable=self.resolution_var, width=8).grid(row=1, column=1, padx=2)
        
        # Action buttons
        action_frame = ttk.Frame(params_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Generate Enhanced Heatmap", 
                  command=self.generate_enhanced_heatmap).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Save Project", 
                  command=self.save_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Load Project", 
                  command=self.load_project).pack(side=tk.LEFT, padx=2)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(parent, text="Floor Plan Designer")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create floor plan canvas
        self.floor_plan_canvas = FloorPlanCanvas(canvas_frame, width=600, height=500)
        
    def setup_visualization_panel(self, parent):
        """Setup the enhanced visualization panel"""
        # Create the enhanced 3D viewer
        self.enhanced_viewer = Enhanced3DViewer(parent)
        
    def on_mode_change(self):
        """Handle drawing mode change"""
        if self.floor_plan_canvas:
            self.floor_plan_canvas.set_drawing_mode(self.drawing_mode.get())
    
    def clear_all(self):
        """Clear all elements"""
        if self.floor_plan_canvas:
            self.floor_plan_canvas.clear_all()
    
    def auto_place_aps(self):
        """Automatically place access points using the algorithm"""
        if not self.floor_plan_canvas:
            return
        
        try:
            # Import the placement function
            from main_four_ap import _place_aps_intelligent_grid, estimate_initial_ap_count
            
            # Get building bounds
            min_x, min_y, max_x, max_y = self.floor_plan_canvas.get_building_bounds()
            width = max_x - min_x
            length = max_y - min_y
            height = self.height_var.get()
            
            # Estimate number of APs needed
            num_aps = estimate_initial_ap_count(width, length, height)
            
            # Create materials grid (simplified)
            materials_grid = self.create_materials_grid(width, length)
            
            # Place APs using the intelligent grid algorithm
            ap_locations = _place_aps_intelligent_grid(
                num_aps=num_aps,
                building_width=width,
                building_length=length,
                building_height=height,
                materials_grid=materials_grid,
                min_ap_sep=7.0,
                min_wall_gap=1.0
            )
            
            # Clear existing APs and add new ones
            self.floor_plan_canvas.access_points.clear()
            
            for ap_name, (x, y, z) in ap_locations.items():
                # Adjust coordinates to floor plan coordinate system
                adjusted_x = min_x + x
                adjusted_y = min_y + y
                self.floor_plan_canvas.place_access_point(adjusted_x, adjusted_y, z)
            
            messagebox.showinfo("Success", f"Placed {len(ap_locations)} access points automatically!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-place APs: {str(e)}")
    
    def create_materials_grid(self, width, length, resolution=0.2):
        """Create a simplified materials grid from floor plan elements"""
        grid_width = int(width / resolution)
        grid_length = int(length / resolution)
        
        # Initialize with air
        materials_grid = [['air' for _ in range(grid_width)] for _ in range(grid_length)]
        
        # Add walls and obstacles (simplified)
        for element in self.floor_plan_canvas.elements:
            if element.type in ['wall', 'obstacle']:
                # Convert element to grid cells (simplified)
                for point in element.points:
                    grid_x = int(point[0] / resolution)
                    grid_y = int(point[1] / resolution)
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_length:
                        materials_grid[grid_y][grid_x] = element.material
        
        return materials_grid
    
    def generate_enhanced_heatmap(self):
        """Generate enhanced heatmap using the 3D viewer"""
        if not self.floor_plan_canvas.access_points:
            messagebox.showwarning("Warning", "No access points placed!")
            return
        
        try:
            # Generate the enhanced heatmap
            resolution = self.resolution_var.get()
            self.enhanced_viewer.generate_enhanced_heatmap(self.floor_plan_canvas, resolution)
            messagebox.showinfo("Success", "Enhanced heatmap generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate heatmap: {str(e)}")
    
    def save_project(self):
        """Save the complete project"""
        if not self.floor_plan_canvas:
            return
        
        from tkinter import filedialog
        import json
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                data = {
                    'elements': [
                        {
                            'type': elem.type,
                            'points': elem.points,
                            'material': elem.material,
                            'height': elem.height,
                            'color': elem.color
                        }
                        for elem in self.floor_plan_canvas.elements
                    ],
                    'access_points': [
                        {
                            'x': ap.x,
                            'y': ap.y,
                            'z': ap.z,
                            'name': ap.name,
                            'power': ap.power,
                            'frequency': ap.frequency,
                            'color': ap.color
                        }
                        for ap in self.floor_plan_canvas.access_points
                    ],
                    'parameters': {
                        'building_height': self.height_var.get(),
                        'target_coverage': self.coverage_var.get(),
                        'resolution': self.resolution_var.get()
                    },
                    'version': '2.0'
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                messagebox.showinfo("Success", "Project saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save project: {str(e)}")
    
    def load_project(self):
        """Load a complete project"""
        if not self.floor_plan_canvas:
            return
        
        from tkinter import filedialog
        import json
        
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Clear existing elements
                self.floor_plan_canvas.clear_all()
                
                # Load elements
                from wifi_gui_app import FloorPlanElement, AccessPoint
                
                for elem_data in data.get('elements', []):
                    element = FloorPlanElement(
                        type=elem_data['type'],
                        points=elem_data['points'],
                        material=elem_data.get('material', 'concrete'),
                        height=elem_data.get('height', 3.0),
                        color=elem_data.get('color', '#000000')
                    )
                    self.floor_plan_canvas.elements.append(element)
                
                # Load access points
                for ap_data in data.get('access_points', []):
                    ap = AccessPoint(
                        x=ap_data['x'],
                        y=ap_data['y'],
                        z=ap_data['z'],
                        name=ap_data['name'],
                        power=ap_data.get('power', 20.0),
                        frequency=ap_data.get('frequency', 2.4),
                        color=ap_data.get('color', '#FF0000')
                    )
                    self.floor_plan_canvas.access_points.append(ap)
                
                # Load parameters
                params = data.get('parameters', {})
                self.height_var.set(params.get('building_height', 3.0))
                self.coverage_var.set(params.get('target_coverage', 90.0))
                self.resolution_var.set(params.get('resolution', 100))
                
                # Redraw everything
                self.floor_plan_canvas.redraw_all()
                
                messagebox.showinfo("Success", "Project loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load project: {str(e)}")
    
    def run(self):
        """Run the integrated application"""
        self.root.mainloop()

def main():
    """Main function to launch the integrated GUI"""
    try:
        # Check if required packages are available
        import numpy
        import matplotlib
        print("✓ All required packages are available")
        
        # Launch the integrated GUI
        app = IntegratedWiFiGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nPlease install required packages:")
        print("pip install -r gui/requirements_gui.txt")
        return 1
    
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
