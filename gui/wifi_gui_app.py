#!/usr/bin/env python3
"""
WiFi AP Placement GUI Application
Interactive Canvas-based floor plan drawing with 2D to 3D simulation and heatmap generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main_four_ap import (
        _place_aps_intelligent_grid, 
        evaluate_coverage_and_capacity,
        estimate_initial_ap_count
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import AP placement functions: {e}")
    BACKEND_AVAILABLE = False

@dataclass
class FloorPlanElement:
    """Represents a floor plan element (wall, room, obstacle)"""
    type: str  # 'wall', 'room', 'obstacle', 'door', 'window'
    points: List[Tuple[float, float]]
    material: str = 'concrete'
    height: float = 3.0
    color: str = '#000000'

@dataclass
class AccessPoint:
    """Represents an access point"""
    x: float
    y: float
    z: float
    name: str
    power: float = 20.0  # dBm
    frequency: float = 2.4  # GHz
    color: str = '#FF0000'

class FloorPlanCanvas:
    """Interactive canvas for drawing floor plans"""
    
    def __init__(self, parent, width=800, height=600):
        self.parent = parent
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Drawing state
        self.drawing_mode = 'wall'
        self.current_element = None
        self.elements = []
        self.access_points = []
        self.scale = 10  # pixels per meter
        self.grid_size = 1.0  # meters
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)
        
        self.draw_grid()
    
    def draw_grid(self):
        """Draw grid lines on canvas"""
        self.canvas.delete("grid")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        grid_pixels = self.grid_size * self.scale
        
        # Vertical lines
        for x in range(0, width, int(grid_pixels)):
            self.canvas.create_line(x, 0, x, height, fill='#E0E0E0', tags="grid")
        
        # Horizontal lines
        for y in range(0, height, int(grid_pixels)):
            self.canvas.create_line(0, y, width, y, fill='#E0E0E0', tags="grid")
    
    def set_drawing_mode(self, mode):
        """Set the current drawing mode"""
        self.drawing_mode = mode
    
    def on_click(self, event):
        """Handle mouse click events"""
        x, y = self.canvas_to_world(event.x, event.y)
        
        if self.drawing_mode == 'ap':
            self.place_access_point(x, y)
        elif self.drawing_mode in ['wall', 'room', 'obstacle']:
            self.start_drawing_element(x, y)
    
    def on_drag(self, event):
        """Handle mouse drag events"""
        if self.current_element:
            x, y = self.canvas_to_world(event.x, event.y)
            self.update_current_element(x, y)
    
    def on_release(self, event):
        """Handle mouse release events"""
        if self.current_element:
            self.finish_current_element()
    
    def on_motion(self, event):
        """Handle mouse motion for preview"""
        pass
    
    def canvas_to_world(self, canvas_x, canvas_y):
        """Convert canvas coordinates to world coordinates (meters)"""
        world_x = canvas_x / self.scale
        world_y = canvas_y / self.scale
        return world_x, world_y
    
    def world_to_canvas(self, world_x, world_y):
        """Convert world coordinates to canvas coordinates"""
        canvas_x = world_x * self.scale
        canvas_y = world_y * self.scale
        return int(canvas_x), int(canvas_y)
    
    def start_drawing_element(self, x, y):
        """Start drawing a new floor plan element"""
        self.current_element = {
            'type': self.drawing_mode,
            'points': [(x, y)],
            'canvas_id': None
        }
    
    def update_current_element(self, x, y):
        """Update the current element being drawn"""
        if self.current_element:
            if len(self.current_element['points']) == 1:
                self.current_element['points'].append((x, y))
            else:
                self.current_element['points'][-1] = (x, y)
            
            self.redraw_current_element()
    
    def finish_current_element(self):
        """Finish drawing the current element"""
        if self.current_element and len(self.current_element['points']) >= 2:
            element = FloorPlanElement(
                type=self.current_element['type'],
                points=self.current_element['points'].copy(),
                material='concrete' if self.current_element['type'] == 'wall' else 'air'
            )
            self.elements.append(element)
            self.redraw_all()
        
        self.current_element = None
    
    def place_access_point(self, x, y, z=2.7):
        """Place an access point at the specified location"""
        ap_name = f"AP{len(self.access_points) + 1}"
        ap = AccessPoint(x=x, y=y, z=z, name=ap_name)
        self.access_points.append(ap)
        self.redraw_all()
    
    def redraw_current_element(self):
        """Redraw the current element being drawn"""
        if self.current_element and self.current_element['canvas_id']:
            self.canvas.delete(self.current_element['canvas_id'])
        
        if self.current_element and len(self.current_element['points']) >= 2:
            points = []
            for point in self.current_element['points']:
                canvas_point = self.world_to_canvas(point[0], point[1])
                points.extend(canvas_point)
            
            color = self.get_element_color(self.current_element['type'])
            self.current_element['canvas_id'] = self.canvas.create_line(
                points, fill=color, width=3, tags="current"
            )
    
    def redraw_all(self):
        """Redraw all elements on the canvas"""
        self.canvas.delete("element")
        self.canvas.delete("ap")
        
        # Draw floor plan elements
        for element in self.elements:
            self.draw_element(element)
        
        # Draw access points
        for ap in self.access_points:
            self.draw_access_point(ap)
    
    def draw_element(self, element):
        """Draw a floor plan element"""
        if len(element.points) < 2:
            return
        
        points = []
        for point in element.points:
            canvas_point = self.world_to_canvas(point[0], point[1])
            points.extend(canvas_point)
        
        color = self.get_element_color(element.type)
        width = 5 if element.type == 'wall' else 2
        
        self.canvas.create_line(points, fill=color, width=width, tags="element")
    
    def draw_access_point(self, ap):
        """Draw an access point"""
        canvas_x, canvas_y = self.world_to_canvas(ap.x, ap.y)
        radius = 8
        
        # Draw AP circle
        self.canvas.create_oval(
            canvas_x - radius, canvas_y - radius,
            canvas_x + radius, canvas_y + radius,
            fill=ap.color, outline='black', width=2, tags="ap"
        )
        
        # Draw AP label
        self.canvas.create_text(
            canvas_x, canvas_y - radius - 15,
            text=ap.name, fill='black', font=('Arial', 8, 'bold'), tags="ap"
        )
    
    def get_element_color(self, element_type):
        """Get color for element type"""
        colors = {
            'wall': '#000000',
            'room': '#0000FF',
            'obstacle': '#FF8000',
            'door': '#00FF00',
            'window': '#00FFFF'
        }
        return colors.get(element_type, '#000000')
    
    def clear_all(self):
        """Clear all elements"""
        self.elements.clear()
        self.access_points.clear()
        self.canvas.delete("element")
        self.canvas.delete("ap")
    
    def get_building_bounds(self):
        """Get the bounds of the building"""
        if not self.elements:
            return 0, 0, 40, 30  # Default bounds
        
        all_points = []
        for element in self.elements:
            all_points.extend(element.points)
        
        if not all_points:
            return 0, 0, 40, 30
        
        xs, ys = zip(*all_points)
        return min(xs), min(ys), max(xs), max(ys)

class HeatmapViewer:
    """3D heatmap visualization component"""
    
    def __init__(self, parent):
        self.parent = parent
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.heatmap_data = None
        self.building_bounds = None
    
    def generate_heatmap(self, floor_plan_canvas, resolution=50):
        """Generate WiFi signal strength heatmap"""
        if not floor_plan_canvas.access_points:
            messagebox.showwarning("Warning", "No access points placed!")
            return
        
        # Get building bounds
        min_x, min_y, max_x, max_y = floor_plan_canvas.get_building_bounds()
        width = max_x - min_x
        height = max_y - min_y
        
        # Create evaluation grid
        x_points = np.linspace(min_x, max_x, resolution)
        y_points = np.linspace(min_y, max_y, resolution)
        
        # Initialize heatmap data
        signal_strength = np.zeros((len(y_points), len(x_points)))
        
        # Calculate signal strength at each point
        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                max_signal = -100  # dBm
                
                for ap in floor_plan_canvas.access_points:
                    # Calculate distance
                    distance = np.sqrt((x - ap.x)**2 + (y - ap.y)**2 + (1.5 - ap.z)**2)
                    if distance < 0.1:
                        distance = 0.1
                    
                    # Free space path loss
                    fspl = 20 * np.log10(distance) + 20 * np.log10(ap.frequency * 1000) + 32.44
                    
                    # Calculate signal strength
                    signal = ap.power - fspl
                    max_signal = max(max_signal, signal)
                
                signal_strength[i, j] = max_signal
        
        self.heatmap_data = signal_strength
        self.building_bounds = (min_x, min_y, max_x, max_y)
        self.display_heatmap(x_points, y_points, signal_strength)
    
    def display_heatmap(self, x_points, y_points, signal_strength):
        """Display the heatmap"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create heatmap
        im = ax.imshow(signal_strength, extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]], 
                      cmap='RdYlGn', origin='lower', alpha=0.8)
        
        # Add colorbar
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Signal Strength (dBm)')
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('WiFi Signal Strength Heatmap')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

class WiFiGUIApp:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WiFi AP Placement Designer")
        self.root.geometry("1400x900")
        
        self.setup_ui()
        self.floor_plan_canvas = None
        self.heatmap_viewer = None
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for floor plan drawing
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Right panel for heatmap and controls
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
    
    def setup_left_panel(self, parent):
        """Setup the left panel with floor plan canvas"""
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
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(parent, text="Floor Plan Designer")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create floor plan canvas
        self.floor_plan_canvas = FloorPlanCanvas(canvas_frame)
    
    def setup_right_panel(self, parent):
        """Setup the right panel with controls and heatmap"""
        # Controls frame
        controls_frame = ttk.LabelFrame(parent, text="Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Building parameters
        ttk.Label(controls_frame, text="Building Height (m):").pack(anchor=tk.W, padx=5)
        self.height_var = tk.DoubleVar(value=3.0)
        ttk.Entry(controls_frame, textvariable=self.height_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(controls_frame, text="Target Coverage (%):").pack(anchor=tk.W, padx=5)
        self.coverage_var = tk.DoubleVar(value=90.0)
        ttk.Entry(controls_frame, textvariable=self.coverage_var, width=10).pack(anchor=tk.W, padx=5, pady=2)
        
        # Heatmap controls
        ttk.Button(controls_frame, text="Generate Heatmap", 
                  command=self.generate_heatmap).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Save Floor Plan", 
                  command=self.save_floor_plan).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(controls_frame, text="Load Floor Plan", 
                  command=self.load_floor_plan).pack(fill=tk.X, padx=5, pady=2)
        
        # Heatmap viewer frame
        heatmap_frame = ttk.LabelFrame(parent, text="WiFi Heatmap")
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create heatmap viewer
        self.heatmap_viewer = HeatmapViewer(heatmap_frame)
    
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
        # This is a simplified implementation
        # In a full implementation, you would convert the drawn elements to a materials grid
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
    
    def generate_heatmap(self):
        """Generate and display WiFi heatmap"""
        if self.heatmap_viewer and self.floor_plan_canvas:
            self.heatmap_viewer.generate_heatmap(self.floor_plan_canvas)
    
    def save_floor_plan(self):
        """Save floor plan to JSON file"""
        if not self.floor_plan_canvas:
            return
        
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
                    'building_height': self.height_var.get(),
                    'target_coverage': self.coverage_var.get()
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                messagebox.showinfo("Success", "Floor plan saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save floor plan: {str(e)}")
    
    def load_floor_plan(self):
        """Load floor plan from JSON file"""
        if not self.floor_plan_canvas:
            return
        
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
                self.height_var.set(data.get('building_height', 3.0))
                self.coverage_var.set(data.get('target_coverage', 90.0))
                
                # Redraw everything
                self.floor_plan_canvas.redraw_all()
                
                messagebox.showinfo("Success", "Floor plan loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load floor plan: {str(e)}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = WiFiGUIApp()
    app.run()
