#!/usr/bin/env python3
"""
Simplified WiFi AP Placement GUI
Reliable version that works on macOS without complex dependencies
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class SimpleFloorPlanCanvas:
    """Simplified floor plan canvas"""
    
    def __init__(self, parent, width=600, height=400):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.drawing_mode = 'wall'
        self.elements = []
        self.access_points = []
        self.scale = 10  # pixels per meter
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.current_line = None
        self.start_x = None
        self.start_y = None
        
        self.draw_grid()
    
    def draw_grid(self):
        """Draw grid"""
        width = 600
        height = 400
        for x in range(0, width, 20):
            self.canvas.create_line(x, 0, x, height, fill='#E0E0E0', tags="grid")
        for y in range(0, height, 20):
            self.canvas.create_line(0, y, width, y, fill='#E0E0E0', tags="grid")
    
    def set_drawing_mode(self, mode):
        self.drawing_mode = mode
    
    def on_click(self, event):
        if self.drawing_mode == 'ap':
            self.place_ap(event.x, event.y)
        else:
            self.start_x = event.x
            self.start_y = event.y
    
    def on_drag(self, event):
        if self.start_x and self.start_y and self.drawing_mode != 'ap':
            if self.current_line:
                self.canvas.delete(self.current_line)
            
            color = 'black' if self.drawing_mode == 'wall' else 'blue'
            width = 3 if self.drawing_mode == 'wall' else 2
            
            self.current_line = self.canvas.create_line(
                self.start_x, self.start_y, event.x, event.y,
                fill=color, width=width, tags="element"
            )
    
    def on_release(self, event):
        if self.start_x and self.start_y and self.drawing_mode != 'ap':
            # Convert to world coordinates
            x1, y1 = self.start_x / self.scale, self.start_y / self.scale
            x2, y2 = event.x / self.scale, event.y / self.scale
            
            element = {
                'type': self.drawing_mode,
                'points': [(x1, y1), (x2, y2)],
                'canvas_id': self.current_line
            }
            self.elements.append(element)
        
        self.current_line = None
        self.start_x = None
        self.start_y = None
    
    def place_ap(self, canvas_x, canvas_y):
        """Place access point"""
        x, y = canvas_x / self.scale, canvas_y / self.scale
        ap_name = f"AP{len(self.access_points) + 1}"
        
        # Draw AP
        ap_id = self.canvas.create_oval(
            canvas_x - 8, canvas_y - 8, canvas_x + 8, canvas_y + 8,
            fill='red', outline='black', width=2, tags="ap"
        )
        
        text_id = self.canvas.create_text(
            canvas_x, canvas_y - 20, text=ap_name,
            fill='black', font=('Arial', 8, 'bold'), tags="ap"
        )
        
        ap = {
            'name': ap_name,
            'x': x, 'y': y, 'z': 2.7,
            'power': 20.0, 'frequency': 2.4,
            'canvas_ids': [ap_id, text_id]
        }
        self.access_points.append(ap)
    
    def clear_all(self):
        """Clear all elements"""
        self.canvas.delete("element")
        self.canvas.delete("ap")
        self.elements.clear()
        self.access_points.clear()
    
    def get_building_bounds(self):
        """Get building bounds"""
        if not self.elements:
            return 0, 0, 40, 30
        
        all_points = []
        for element in self.elements:
            all_points.extend(element['points'])
        
        if not all_points:
            return 0, 0, 40, 30
        
        xs, ys = zip(*all_points)
        return min(xs), min(ys), max(xs), max(ys)

class SimpleHeatmapViewer:
    """Simplified heatmap viewer"""
    
    def __init__(self, parent):
        self.figure = Figure(figsize=(8, 6), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_heatmap(self, floor_plan_canvas):
        """Generate simple heatmap"""
        if not floor_plan_canvas.access_points:
            messagebox.showwarning("Warning", "No access points placed!")
            return
        
        # Get bounds
        min_x, min_y, max_x, max_y = floor_plan_canvas.get_building_bounds()
        
        # Create grid
        resolution = 50
        x_points = np.linspace(min_x, max_x, resolution)
        y_points = np.linspace(min_y, max_y, resolution)
        
        signal_strength = np.zeros((len(y_points), len(x_points)))
        
        # Calculate signal strength
        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                max_signal = -100
                
                for ap in floor_plan_canvas.access_points:
                    distance = np.sqrt((x - ap['x'])**2 + (y - ap['y'])**2 + (1.5 - ap['z'])**2)
                    if distance < 0.1:
                        distance = 0.1
                    
                    # Simple path loss
                    fspl = 20 * np.log10(distance) + 20 * np.log10(ap['frequency'] * 1000) + 32.44
                    signal = ap['power'] - fspl
                    max_signal = max(max_signal, signal)
                
                signal_strength[i, j] = max_signal
        
        # Plot heatmap
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        im = ax.imshow(signal_strength, extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]], 
                      cmap='RdYlGn', origin='lower', alpha=0.8, vmin=-100, vmax=-30)
        
        # Draw APs
        for ap in floor_plan_canvas.access_points:
            ax.plot(ap['x'], ap['y'], 'ro', markersize=8, markeredgecolor='black')
            ax.text(ap['x'], ap['y'] + 1, ap['name'], ha='center', fontsize=8, fontweight='bold')
        
        # Draw walls
        for element in floor_plan_canvas.elements:
            if element['type'] == 'wall':
                points = element['points']
                if len(points) >= 2:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, 'k-', linewidth=3, alpha=0.8)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('WiFi Signal Strength Heatmap')
        
        # Colorbar
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Signal Strength (dBm)')
        
        self.canvas.draw()

class SimpleWiFiGUI:
    """Simplified WiFi GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WiFi AP Placement Designer - Simple")
        self.root.geometry("1200x800")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        # Main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
    
    def setup_left_panel(self, parent):
        """Setup drawing panel"""
        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=5)
        
        self.drawing_mode = tk.StringVar(value='wall')
        modes = [('Wall', 'wall'), ('Room', 'room'), ('AP', 'ap')]
        
        for text, mode in modes:
            ttk.Radiobutton(toolbar, text=text, variable=self.drawing_mode, 
                           value=mode, command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Clear", command=self.clear_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="Auto APs", command=self.auto_place_aps).pack(side=tk.RIGHT, padx=2)
        
        # Canvas
        canvas_frame = ttk.LabelFrame(parent, text="Floor Plan")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.floor_plan = SimpleFloorPlanCanvas(canvas_frame)
    
    def setup_right_panel(self, parent):
        """Setup visualization panel"""
        # Controls
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls, text="Generate Heatmap", 
                  command=self.generate_heatmap).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Save", command=self.save_project).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Load", command=self.load_project).pack(side=tk.LEFT, padx=5)
        
        # Heatmap
        heatmap_frame = ttk.LabelFrame(parent, text="Heatmap")
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.heatmap_viewer = SimpleHeatmapViewer(heatmap_frame)
    
    def on_mode_change(self):
        """Handle mode change"""
        self.floor_plan.set_drawing_mode(self.drawing_mode.get())
    
    def clear_all(self):
        """Clear all"""
        self.floor_plan.clear_all()
    
    def auto_place_aps(self):
        """Auto place APs"""
        try:
            # Simple auto placement - place APs in a grid
            min_x, min_y, max_x, max_y = self.floor_plan.get_building_bounds()
            width = max_x - min_x
            height = max_y - min_y
            
            # Clear existing APs
            self.floor_plan.canvas.delete("ap")
            self.floor_plan.access_points.clear()
            
            # Place APs in a 2x2 grid
            for i in range(2):
                for j in range(2):
                    x = min_x + (i + 0.5) * width / 2
                    y = min_y + (j + 0.5) * height / 2
                    canvas_x = x * self.floor_plan.scale
                    canvas_y = y * self.floor_plan.scale
                    self.floor_plan.place_ap(canvas_x, canvas_y)
            
            messagebox.showinfo("Success", f"Placed {len(self.floor_plan.access_points)} APs")
            
        except Exception as e:
            messagebox.showerror("Error", f"Auto placement failed: {e}")
    
    def generate_heatmap(self):
        """Generate heatmap"""
        self.heatmap_viewer.generate_heatmap(self.floor_plan)
    
    def save_project(self):
        """Save project"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                data = {
                    'elements': [
                        {
                            'type': elem['type'],
                            'points': elem['points']
                        }
                        for elem in self.floor_plan.elements
                    ],
                    'access_points': [
                        {
                            'name': ap['name'],
                            'x': ap['x'], 'y': ap['y'], 'z': ap['z'],
                            'power': ap['power'], 'frequency': ap['frequency']
                        }
                        for ap in self.floor_plan.access_points
                    ]
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                messagebox.showinfo("Success", "Project saved!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")
    
    def load_project(self):
        """Load project"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Clear existing
                self.floor_plan.clear_all()
                
                # Load elements
                for elem_data in data.get('elements', []):
                    element = {
                        'type': elem_data['type'],
                        'points': elem_data['points'],
                        'canvas_id': None
                    }
                    
                    # Redraw element
                    if len(element['points']) >= 2:
                        x1, y1 = element['points'][0]
                        x2, y2 = element['points'][1]
                        
                        canvas_x1, canvas_y1 = x1 * self.floor_plan.scale, y1 * self.floor_plan.scale
                        canvas_x2, canvas_y2 = x2 * self.floor_plan.scale, y2 * self.floor_plan.scale
                        
                        color = 'black' if element['type'] == 'wall' else 'blue'
                        width = 3 if element['type'] == 'wall' else 2
                        
                        canvas_id = self.floor_plan.canvas.create_line(
                            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                            fill=color, width=width, tags="element"
                        )
                        element['canvas_id'] = canvas_id
                    
                    self.floor_plan.elements.append(element)
                
                # Load APs
                for ap_data in data.get('access_points', []):
                    canvas_x = ap_data['x'] * self.floor_plan.scale
                    canvas_y = ap_data['y'] * self.floor_plan.scale
                    self.floor_plan.place_ap(canvas_x, canvas_y)
                
                messagebox.showinfo("Success", "Project loaded!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Load failed: {e}")
    
    def run(self):
        """Run the app"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = SimpleWiFiGUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
