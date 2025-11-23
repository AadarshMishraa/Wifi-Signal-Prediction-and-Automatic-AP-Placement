#!/usr/bin/env python3
"""
Working WiFi AP Placement GUI - Optimized for immediate startup
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class WorkingFloorPlanCanvas:
    """Working floor plan canvas with immediate response"""
    
    def __init__(self, parent, width=700, height=500):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='white', relief=tk.SUNKEN, bd=2)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # State
        self.drawing_mode = 'wall'
        self.elements = []
        self.access_points = []
        self.scale = 15  # pixels per meter
        
        # Drawing state
        self.current_line = None
        self.start_x = None
        self.start_y = None
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.draw_grid()
        self.update_status("Ready to draw. Select a mode and start drawing.")
    
    def draw_grid(self):
        """Draw grid lines"""
        self.canvas.delete("grid")
        width = 700
        height = 500
        
        # Grid every 20 pixels (approximately 1.33 meters)
        for x in range(0, width, 20):
            self.canvas.create_line(x, 0, x, height, fill='#E8E8E8', tags="grid")
        for y in range(0, height, 20):
            self.canvas.create_line(0, y, width, y, fill='#E8E8E8', tags="grid")
    
    def set_drawing_mode(self, mode):
        """Set drawing mode"""
        self.drawing_mode = mode
        if mode == 'wall':
            self.update_status("Wall mode: Click and drag to draw walls")
        elif mode == 'room':
            self.update_status("Room mode: Click and drag to draw room boundaries")
        elif mode == 'obstacle':
            self.update_status("Obstacle mode: Click and drag to draw obstacles")
        elif mode == 'ap':
            self.update_status("AP mode: Click to place access points")
    
    def update_status(self, message):
        """Update status (placeholder for now)"""
        print(f"Status: {message}")
    
    def on_click(self, event):
        """Handle mouse click"""
        if self.drawing_mode == 'ap':
            self.place_access_point(event.x, event.y)
        else:
            self.start_x = event.x
            self.start_y = event.y
    
    def on_drag(self, event):
        """Handle mouse drag"""
        if self.start_x and self.start_y and self.drawing_mode != 'ap':
            # Remove previous preview line
            if self.current_line:
                self.canvas.delete(self.current_line)
            
            # Draw preview line
            color = self.get_color_for_mode(self.drawing_mode)
            width = self.get_width_for_mode(self.drawing_mode)
            
            self.current_line = self.canvas.create_line(
                self.start_x, self.start_y, event.x, event.y,
                fill=color, width=width, tags="preview"
            )
    
    def on_release(self, event):
        """Handle mouse release"""
        if self.start_x and self.start_y and self.drawing_mode != 'ap':
            # Convert to world coordinates
            x1, y1 = self.start_x / self.scale, self.start_y / self.scale
            x2, y2 = event.x / self.scale, event.y / self.scale
            
            # Create element
            element = {
                'type': self.drawing_mode,
                'points': [(x1, y1), (x2, y2)],
                'material': 'concrete' if self.drawing_mode == 'wall' else 'air'
            }
            self.elements.append(element)
            
            # Draw final line
            color = self.get_color_for_mode(self.drawing_mode)
            width = self.get_width_for_mode(self.drawing_mode)
            
            line_id = self.canvas.create_line(
                self.start_x, self.start_y, event.x, event.y,
                fill=color, width=width, tags="element"
            )
            element['canvas_id'] = line_id
            
            self.update_status(f"Drew {self.drawing_mode}: {len(self.elements)} elements total")
        
        # Reset drawing state
        if self.current_line:
            self.canvas.delete(self.current_line)
        self.current_line = None
        self.start_x = None
        self.start_y = None
    
    def place_access_point(self, canvas_x, canvas_y):
        """Place an access point"""
        # Convert to world coordinates
        x, y = canvas_x / self.scale, canvas_y / self.scale
        z = 2.7  # Default ceiling height
        
        ap_name = f"AP{len(self.access_points) + 1}"
        
        # Draw AP on canvas
        radius = 8
        ap_circle = self.canvas.create_oval(
            canvas_x - radius, canvas_y - radius,
            canvas_x + radius, canvas_y + radius,
            fill='red', outline='darkred', width=2, tags="ap"
        )
        
        ap_text = self.canvas.create_text(
            canvas_x, canvas_y - radius - 15,
            text=ap_name, fill='black', font=('Arial', 9, 'bold'), tags="ap"
        )
        
        # Store AP data
        ap = {
            'name': ap_name,
            'x': x, 'y': y, 'z': z,
            'power': 20.0,  # dBm
            'frequency': 2.4,  # GHz
            'canvas_ids': [ap_circle, ap_text]
        }
        self.access_points.append(ap)
        
        self.update_status(f"Placed {ap_name} at ({x:.1f}, {y:.1f})")
    
    def get_color_for_mode(self, mode):
        """Get color for drawing mode"""
        colors = {
            'wall': 'black',
            'room': 'blue',
            'obstacle': 'orange'
        }
        return colors.get(mode, 'black')
    
    def get_width_for_mode(self, mode):
        """Get line width for drawing mode"""
        widths = {
            'wall': 4,
            'room': 2,
            'obstacle': 3
        }
        return widths.get(mode, 2)
    
    def clear_all(self):
        """Clear all elements"""
        self.canvas.delete("element")
        self.canvas.delete("ap")
        self.canvas.delete("preview")
        self.elements.clear()
        self.access_points.clear()
        self.update_status("Cleared all elements")
    
    def get_building_bounds(self):
        """Get building bounds"""
        if not self.elements:
            return 0, 0, 40, 30  # Default bounds
        
        all_points = []
        for element in self.elements:
            all_points.extend(element['points'])
        
        if not all_points:
            return 0, 0, 40, 30
        
        xs, ys = zip(*all_points)
        return min(xs), min(ys), max(xs), max(ys)

class WorkingHeatmapViewer:
    """Working heatmap viewer with immediate response"""
    
    def __init__(self, parent):
        self.figure = Figure(figsize=(9, 7), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add initial message
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Show welcome message"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'WiFi Heatmap Viewer\n\nPlace some access points and\nclick "Generate Heatmap" to begin',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
    
    def generate_heatmap(self, floor_plan_canvas):
        """Generate WiFi heatmap"""
        if not floor_plan_canvas.access_points:
            messagebox.showwarning("Warning", "No access points placed!\nPlace some APs first.")
            return
        
        try:
            # Get building bounds
            min_x, min_y, max_x, max_y = floor_plan_canvas.get_building_bounds()
            
            # Create evaluation grid
            resolution = 60  # Reduced for faster computation
            x_points = np.linspace(min_x, max_x, resolution)
            y_points = np.linspace(min_y, max_y, resolution)
            
            # Initialize signal strength grid
            signal_strength = np.zeros((len(y_points), len(x_points)))
            
            # Calculate signal strength at each point
            for i, y in enumerate(y_points):
                for j, x in enumerate(x_points):
                    max_signal = -100  # dBm
                    
                    for ap in floor_plan_canvas.access_points:
                        # Calculate 3D distance
                        distance = np.sqrt((x - ap['x'])**2 + (y - ap['y'])**2 + (1.5 - ap['z'])**2)
                        if distance < 0.1:
                            distance = 0.1  # Avoid division by zero
                        
                        # Simple free space path loss model
                        fspl = 20 * np.log10(distance) + 20 * np.log10(ap['frequency'] * 1000) + 32.44
                        
                        # Add wall attenuation (simplified)
                        wall_loss = self.calculate_wall_loss(x, y, ap['x'], ap['y'], floor_plan_canvas.elements)
                        
                        # Calculate received signal strength
                        signal = ap['power'] - fspl - wall_loss
                        max_signal = max(max_signal, signal)
                    
                    signal_strength[i, j] = max_signal
            
            # Plot heatmap
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Create heatmap
            im = ax.imshow(signal_strength, 
                          extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]], 
                          cmap='RdYlGn', origin='lower', alpha=0.8, 
                          vmin=-100, vmax=-30)
            
            # Draw floor plan elements
            for element in floor_plan_canvas.elements:
                if len(element['points']) >= 2:
                    xs, ys = zip(*element['points'])
                    color = 'black' if element['type'] == 'wall' else 'blue'
                    width = 3 if element['type'] == 'wall' else 2
                    ax.plot(xs, ys, color=color, linewidth=width, alpha=0.8)
            
            # Draw access points
            for ap in floor_plan_canvas.access_points:
                ax.plot(ap['x'], ap['y'], 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
                ax.annotate(ap['name'], (ap['x'], ap['y']), xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label('Signal Strength (dBm)', fontsize=10)
            
            # Set labels and title
            ax.set_xlabel('X (meters)', fontsize=10)
            ax.set_ylabel('Y (meters)', fontsize=10)
            ax.set_title('WiFi Signal Strength Heatmap', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Calculate coverage statistics
            coverage_threshold = -70  # dBm
            total_points = signal_strength.size
            covered_points = np.sum(signal_strength >= coverage_threshold)
            coverage_percent = (covered_points / total_points) * 100
            
            # Add coverage info
            ax.text(0.02, 0.98, f'Coverage: {coverage_percent:.1f}%\nAPs: {len(floor_plan_canvas.access_points)}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            self.canvas.draw()
            
            print(f"Heatmap generated: {coverage_percent:.1f}% coverage with {len(floor_plan_canvas.access_points)} APs")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate heatmap:\n{str(e)}")
            print(f"Heatmap generation error: {e}")
    
    def calculate_wall_loss(self, x1, y1, x2, y2, elements):
        """Calculate wall attenuation between two points"""
        total_loss = 0
        
        for element in elements:
            if element['type'] == 'wall' and len(element['points']) >= 2:
                wall_start, wall_end = element['points']
                
                # Simple line intersection check
                if self.lines_intersect(x1, y1, x2, y2, wall_start[0], wall_start[1], wall_end[0], wall_end[1]):
                    total_loss += 10  # 10 dB per wall
        
        return min(total_loss, 40)  # Cap at 40 dB
    
    def lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments intersect"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1

class WorkingWiFiGUI:
    """Working WiFi GUI with immediate startup"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WiFi AP Placement Designer")
        self.root.geometry("1400x900")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Draw walls, place APs, and generate heatmaps")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_ui(self):
        """Setup user interface"""
        # Main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (40% width)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Right panel (60% width)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        self.setup_drawing_panel(left_frame)
        self.setup_visualization_panel(right_frame)
    
    def setup_drawing_panel(self, parent):
        """Setup drawing panel"""
        # Toolbar
        toolbar = ttk.LabelFrame(parent, text="Drawing Tools")
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Drawing mode selection
        mode_frame = ttk.Frame(toolbar)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        
        self.drawing_mode = tk.StringVar(value='wall')
        modes = [
            ('Wall', 'wall'),
            ('Room', 'room'),
            ('Obstacle', 'obstacle'),
            ('Access Point', 'ap')
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.drawing_mode, 
                           value=mode, command=self.on_mode_change).pack(side=tk.LEFT, padx=3)
        
        # Action buttons
        action_frame = ttk.Frame(toolbar)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Auto Place APs", command=self.auto_place_aps).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Save Project", command=self.save_project).pack(side=tk.RIGHT, padx=2)
        ttk.Button(action_frame, text="Load Project", command=self.load_project).pack(side=tk.RIGHT, padx=2)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(parent, text="Floor Plan Designer")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create floor plan canvas
        self.floor_plan = WorkingFloorPlanCanvas(canvas_frame)
    
    def setup_visualization_panel(self, parent):
        """Setup visualization panel"""
        # Controls
        controls_frame = ttk.LabelFrame(parent, text="Visualization Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        controls_grid = ttk.Frame(controls_frame)
        controls_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters
        ttk.Label(controls_grid, text="Building Height (m):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.height_var = tk.DoubleVar(value=3.0)
        ttk.Entry(controls_grid, textvariable=self.height_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(controls_grid, text="Target Coverage (%):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.coverage_var = tk.DoubleVar(value=90.0)
        ttk.Entry(controls_grid, textvariable=self.coverage_var, width=8).grid(row=0, column=3, padx=5)
        
        # Generate button
        ttk.Button(controls_frame, text="🔥 Generate Heatmap", 
                  command=self.generate_heatmap).pack(pady=10)
        
        # Heatmap viewer
        heatmap_frame = ttk.LabelFrame(parent, text="WiFi Signal Heatmap")
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.heatmap_viewer = WorkingHeatmapViewer(heatmap_frame)
    
    def on_mode_change(self):
        """Handle drawing mode change"""
        mode = self.drawing_mode.get()
        self.floor_plan.set_drawing_mode(mode)
        self.status_var.set(f"Mode: {mode.title()} - {'Click to place' if mode == 'ap' else 'Click and drag to draw'}")
    
    def clear_all(self):
        """Clear all elements"""
        self.floor_plan.clear_all()
        self.status_var.set("Cleared all elements")
    
    def auto_place_aps(self):
        """Auto place access points"""
        try:
            # Get building bounds
            min_x, min_y, max_x, max_y = self.floor_plan.get_building_bounds()
            width = max_x - min_x
            height = max_y - min_y
            
            if width < 5 or height < 5:
                messagebox.showwarning("Warning", "Draw some walls first to define the building area!")
                return
            
            # Clear existing APs
            self.floor_plan.canvas.delete("ap")
            self.floor_plan.access_points.clear()
            
            # Simple grid placement
            num_aps_x = max(1, int(width / 15))  # One AP per 15 meters
            num_aps_y = max(1, int(height / 15))
            
            for i in range(num_aps_x):
                for j in range(num_aps_y):
                    x = min_x + (i + 0.5) * width / num_aps_x
                    y = min_y + (j + 0.5) * height / num_aps_y
                    
                    canvas_x = x * self.floor_plan.scale
                    canvas_y = y * self.floor_plan.scale
                    
                    self.floor_plan.place_access_point(canvas_x, canvas_y)
            
            num_placed = len(self.floor_plan.access_points)
            messagebox.showinfo("Success", f"Automatically placed {num_placed} access points!")
            self.status_var.set(f"Auto-placed {num_placed} APs")
            
        except Exception as e:
            messagebox.showerror("Error", f"Auto placement failed: {str(e)}")
    
    def generate_heatmap(self):
        """Generate heatmap"""
        self.status_var.set("Generating heatmap...")
        self.root.update()  # Update UI
        
        self.heatmap_viewer.generate_heatmap(self.floor_plan)
        self.status_var.set("Heatmap generated successfully")
    
    def save_project(self):
        """Save project"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                data = {
                    'elements': [
                        {
                            'type': elem['type'],
                            'points': elem['points'],
                            'material': elem.get('material', 'concrete')
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
                    ],
                    'parameters': {
                        'building_height': self.height_var.get(),
                        'target_coverage': self.coverage_var.get()
                    }
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                messagebox.showinfo("Success", "Project saved successfully!")
                self.status_var.set(f"Project saved: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def load_project(self):
        """Load project"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
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
                        'material': elem_data.get('material', 'concrete')
                    }
                    
                    # Redraw element on canvas
                    if len(element['points']) >= 2:
                        x1, y1 = element['points'][0]
                        x2, y2 = element['points'][1]
                        
                        canvas_x1 = x1 * self.floor_plan.scale
                        canvas_y1 = y1 * self.floor_plan.scale
                        canvas_x2 = x2 * self.floor_plan.scale
                        canvas_y2 = y2 * self.floor_plan.scale
                        
                        color = self.floor_plan.get_color_for_mode(element['type'])
                        width = self.floor_plan.get_width_for_mode(element['type'])
                        
                        line_id = self.floor_plan.canvas.create_line(
                            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                            fill=color, width=width, tags="element"
                        )
                        element['canvas_id'] = line_id
                    
                    self.floor_plan.elements.append(element)
                
                # Load APs
                for ap_data in data.get('access_points', []):
                    canvas_x = ap_data['x'] * self.floor_plan.scale
                    canvas_y = ap_data['y'] * self.floor_plan.scale
                    self.floor_plan.place_access_point(canvas_x, canvas_y)
                
                # Load parameters
                params = data.get('parameters', {})
                self.height_var.set(params.get('building_height', 3.0))
                self.coverage_var.set(params.get('target_coverage', 90.0))
                
                messagebox.showinfo("Success", "Project loaded successfully!")
                self.status_var.set(f"Project loaded: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Load failed: {str(e)}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting WiFi AP Placement Designer...")
    
    try:
        app = WorkingWiFiGUI()
        print("GUI created successfully, starting...")
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
