#!/usr/bin/env python3
"""
Enhanced 3D Visualization for WiFi AP Placement
Provides 3D floor plan visualization with interactive heatmaps and AP placement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Tuple, Optional
import math

class Enhanced3DViewer:
    """Enhanced 3D visualization component for WiFi heatmaps and floor plans"""
    
    def __init__(self, parent):
        self.parent = parent
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 2D Heatmap tab
        self.heatmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="2D Heatmap")
        
        # 3D Visualization tab
        self.view3d_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.view3d_frame, text="3D View")
        
        # Coverage Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Coverage Analysis")
        
        self.setup_2d_viewer()
        self.setup_3d_viewer()
        self.setup_analysis_viewer()
        
        # Data storage
        self.floor_plan_data = None
        self.heatmap_data = None
        self.building_bounds = None
        self.access_points = []
        
    def setup_2d_viewer(self):
        """Setup 2D heatmap viewer"""
        self.figure_2d = Figure(figsize=(10, 8), dpi=100)
        self.canvas_2d = FigureCanvasTkAgg(self.figure_2d, self.heatmap_frame)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls for 2D view
        controls_2d = ttk.Frame(self.heatmap_frame)
        controls_2d.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_2d, text="Heatmap Type:").pack(side=tk.LEFT, padx=5)
        self.heatmap_type = tk.StringVar(value="signal_strength")
        heatmap_options = [
            ("Signal Strength", "signal_strength"),
            ("Coverage", "coverage"),
            ("Interference", "interference"),
            ("SINR", "sinr")
        ]
        
        for text, value in heatmap_options:
            ttk.Radiobutton(controls_2d, text=text, variable=self.heatmap_type, 
                           value=value, command=self.update_2d_view).pack(side=tk.LEFT, padx=2)
    
    def setup_3d_viewer(self):
        """Setup 3D visualization viewer"""
        self.figure_3d = Figure(figsize=(10, 8), dpi=100)
        self.canvas_3d = FigureCanvasTkAgg(self.figure_3d, self.view3d_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls for 3D view
        controls_3d = ttk.Frame(self.view3d_frame)
        controls_3d.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_3d, text="View Mode:").pack(side=tk.LEFT, padx=5)
        self.view_mode = tk.StringVar(value="isometric")
        view_options = [
            ("Isometric", "isometric"),
            ("Top View", "top"),
            ("Side View", "side"),
            ("Interactive", "interactive")
        ]
        
        for text, value in view_options:
            ttk.Radiobutton(controls_3d, text=text, variable=self.view_mode, 
                           value=value, command=self.update_3d_view).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(controls_3d, text="Rotate View", 
                  command=self.rotate_3d_view).pack(side=tk.RIGHT, padx=5)
    
    def setup_analysis_viewer(self):
        """Setup coverage analysis viewer"""
        self.figure_analysis = Figure(figsize=(12, 8), dpi=100)
        self.canvas_analysis = FigureCanvasTkAgg(self.figure_analysis, self.analysis_frame)
        self.canvas_analysis.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Analysis controls
        controls_analysis = ttk.Frame(self.analysis_frame)
        controls_analysis.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_analysis, text="Generate Report", 
                  command=self.generate_coverage_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_analysis, text="Export Data", 
                  command=self.export_analysis_data).pack(side=tk.LEFT, padx=5)
    
    def generate_enhanced_heatmap(self, floor_plan_canvas, resolution=100):
        """Generate enhanced WiFi heatmap with multiple metrics"""
        if not floor_plan_canvas.access_points:
            return
        
        # Get building bounds
        min_x, min_y, max_x, max_y = floor_plan_canvas.get_building_bounds()
        width = max_x - min_x
        height = max_y - min_y
        
        # Create evaluation grid
        x_points = np.linspace(min_x, max_x, resolution)
        y_points = np.linspace(min_y, max_y, resolution)
        
        # Initialize data arrays
        signal_strength = np.zeros((len(y_points), len(x_points)))
        coverage_map = np.zeros((len(y_points), len(x_points)))
        interference_map = np.zeros((len(y_points), len(x_points)))
        sinr_map = np.zeros((len(y_points), len(x_points)))
        
        # Calculate metrics at each point
        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                signals = []
                
                # Calculate signal from each AP
                for ap in floor_plan_canvas.access_points:
                    distance_3d = np.sqrt((x - ap.x)**2 + (y - ap.y)**2 + (1.5 - ap.z)**2)
                    if distance_3d < 0.1:
                        distance_3d = 0.1
                    
                    # Enhanced path loss model
                    fspl = self.calculate_path_loss(distance_3d, ap.frequency)
                    wall_loss = self.calculate_wall_attenuation(x, y, ap.x, ap.y, floor_plan_canvas.elements)
                    
                    signal = ap.power - fspl - wall_loss
                    signals.append(signal)
                
                if signals:
                    # Signal strength (strongest signal)
                    max_signal = max(signals)
                    signal_strength[i, j] = max_signal
                    
                    # Coverage (binary: above threshold or not)
                    coverage_map[i, j] = 1 if max_signal >= -70 else 0
                    
                    # Interference (sum of other signals)
                    interference_power = sum(10**(s/10) for s in signals if s != max_signal)
                    interference_db = 10 * np.log10(interference_power) if interference_power > 0 else -100
                    interference_map[i, j] = interference_db
                    
                    # SINR (Signal to Interference plus Noise Ratio)
                    noise_floor = -95  # dBm
                    signal_power = 10**(max_signal/10)
                    noise_power = 10**(noise_floor/10)
                    total_interference = interference_power + noise_power
                    sinr = 10 * np.log10(signal_power / total_interference) if total_interference > 0 else max_signal - noise_floor
                    sinr_map[i, j] = sinr
        
        # Store data
        self.heatmap_data = {
            'signal_strength': signal_strength,
            'coverage': coverage_map,
            'interference': interference_map,
            'sinr': sinr_map,
            'x_points': x_points,
            'y_points': y_points
        }
        self.building_bounds = (min_x, min_y, max_x, max_y)
        self.access_points = floor_plan_canvas.access_points.copy()
        self.floor_plan_data = floor_plan_canvas.elements.copy()
        
        # Update all views
        self.update_2d_view()
        self.update_3d_view()
        self.update_analysis_view()
    
    def calculate_path_loss(self, distance, frequency):
        """Calculate path loss using enhanced model"""
        if distance < 1.0:
            # Close-in reference distance model
            return 20 * np.log10(frequency * 1000) + 20 * np.log10(distance) + 32.44
        else:
            # Two-ray ground reflection model for larger distances
            return 40 * np.log10(distance) + 20 * np.log10(frequency * 1000) - 10
    
    def calculate_wall_attenuation(self, x1, y1, x2, y2, elements):
        """Calculate wall attenuation between two points"""
        total_attenuation = 0
        
        for element in elements:
            if element.type == 'wall' and len(element.points) >= 2:
                # Check if line intersects with wall
                if self.line_intersects_wall(x1, y1, x2, y2, element.points):
                    # Add attenuation based on material
                    if element.material == 'concrete':
                        total_attenuation += 15
                    elif element.material == 'drywall':
                        total_attenuation += 8
                    elif element.material == 'glass':
                        total_attenuation += 3
                    else:
                        total_attenuation += 10  # Default wall
        
        return min(total_attenuation, 40)  # Cap at 40 dB
    
    def line_intersects_wall(self, x1, y1, x2, y2, wall_points):
        """Check if line intersects with wall segment"""
        if len(wall_points) < 2:
            return False
        
        for i in range(len(wall_points) - 1):
            wx1, wy1 = wall_points[i]
            wx2, wy2 = wall_points[i + 1]
            
            if self.lines_intersect(x1, y1, x2, y2, wx1, wy1, wx2, wy2):
                return True
        
        return False
    
    def lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments intersect"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def update_2d_view(self):
        """Update 2D heatmap view"""
        if not self.heatmap_data:
            return
        
        self.figure_2d.clear()
        ax = self.figure_2d.add_subplot(111)
        
        # Get selected heatmap type
        heatmap_type = self.heatmap_type.get()
        data = self.heatmap_data[heatmap_type]
        x_points = self.heatmap_data['x_points']
        y_points = self.heatmap_data['y_points']
        
        # Choose colormap and labels based on type
        if heatmap_type == 'signal_strength':
            cmap = 'RdYlGn'
            label = 'Signal Strength (dBm)'
            vmin, vmax = -100, -30
        elif heatmap_type == 'coverage':
            cmap = 'RdYlGn'
            label = 'Coverage (0=No, 1=Yes)'
            vmin, vmax = 0, 1
        elif heatmap_type == 'interference':
            cmap = 'Reds'
            label = 'Interference (dBm)'
            vmin, vmax = -100, -40
        else:  # sinr
            cmap = 'viridis'
            label = 'SINR (dB)'
            vmin, vmax = -10, 30
        
        # Create heatmap
        im = ax.imshow(data, extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]], 
                      cmap=cmap, origin='lower', alpha=0.8, vmin=vmin, vmax=vmax)
        
        # Draw floor plan elements
        self.draw_floor_plan_2d(ax)
        
        # Draw access points
        for ap in self.access_points:
            ax.plot(ap.x, ap.y, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax.annotate(ap.name, (ap.x, ap.y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add colorbar
        cbar = self.figure_2d.colorbar(im, ax=ax)
        cbar.set_label(label)
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'WiFi {heatmap_type.replace("_", " ").title()} Heatmap')
        ax.grid(True, alpha=0.3)
        
        self.canvas_2d.draw()
    
    def update_3d_view(self):
        """Update 3D visualization"""
        if not self.heatmap_data:
            return
        
        self.figure_3d.clear()
        ax = self.figure_3d.add_subplot(111, projection='3d')
        
        # Get data
        signal_data = self.heatmap_data['signal_strength']
        x_points = self.heatmap_data['x_points']
        y_points = self.heatmap_data['y_points']
        
        # Create meshgrid
        X, Y = np.meshgrid(x_points, y_points)
        
        # Create 3D surface
        surf = ax.plot_surface(X, Y, signal_data, cmap='RdYlGn', alpha=0.7)
        
        # Draw floor plan in 3D
        self.draw_floor_plan_3d(ax)
        
        # Draw access points in 3D
        for ap in self.access_points:
            ax.scatter([ap.x], [ap.y], [ap.z], c='red', s=100, marker='o')
            ax.text(ap.x, ap.y, ap.z + 0.5, ap.name, fontsize=8)
        
        # Set view based on mode
        view_mode = self.view_mode.get()
        if view_mode == "top":
            ax.view_init(elev=90, azim=0)
        elif view_mode == "side":
            ax.view_init(elev=0, azim=0)
        elif view_mode == "isometric":
            ax.view_init(elev=30, azim=45)
        
        # Labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Signal Strength (dBm)')
        ax.set_title('3D WiFi Signal Strength Visualization')
        
        # Add colorbar
        self.figure_3d.colorbar(surf, ax=ax, shrink=0.5)
        
        self.canvas_3d.draw()
    
    def update_analysis_view(self):
        """Update coverage analysis view"""
        if not self.heatmap_data:
            return
        
        self.figure_analysis.clear()
        
        # Create subplots for different analyses
        gs = self.figure_analysis.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Coverage statistics
        ax1 = self.figure_analysis.add_subplot(gs[0, 0])
        coverage_data = self.heatmap_data['coverage']
        coverage_percent = np.mean(coverage_data) * 100
        
        labels = ['Covered', 'Not Covered']
        sizes = [coverage_percent, 100 - coverage_percent]
        colors = ['#90EE90', '#FFB6C1']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Coverage Analysis\n{coverage_percent:.1f}% Covered')
        
        # Signal strength distribution
        ax2 = self.figure_analysis.add_subplot(gs[0, 1])
        signal_data = self.heatmap_data['signal_strength'].flatten()
        ax2.hist(signal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(-70, color='red', linestyle='--', label='Coverage Threshold')
        ax2.set_xlabel('Signal Strength (dBm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Signal Strength Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # SINR analysis
        ax3 = self.figure_analysis.add_subplot(gs[1, 0])
        sinr_data = self.heatmap_data['sinr'].flatten()
        ax3.hist(sinr_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(10, color='red', linestyle='--', label='Good Quality Threshold')
        ax3.set_xlabel('SINR (dB)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('SINR Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # AP performance comparison
        ax4 = self.figure_analysis.add_subplot(gs[1, 1])
        ap_names = [ap.name for ap in self.access_points]
        ap_coverage = []
        
        for ap in self.access_points:
            # Calculate coverage area for each AP (simplified)
            x_points = self.heatmap_data['x_points']
            y_points = self.heatmap_data['y_points']
            coverage_count = 0
            total_points = 0
            
            for i, y in enumerate(y_points):
                for j, x in enumerate(x_points):
                    distance = np.sqrt((x - ap.x)**2 + (y - ap.y)**2)
                    if distance <= 15:  # Within AP range
                        total_points += 1
                        if self.heatmap_data['coverage'][i, j] > 0:
                            coverage_count += 1
            
            coverage_ratio = coverage_count / total_points if total_points > 0 else 0
            ap_coverage.append(coverage_ratio * 100)
        
        bars = ax4.bar(ap_names, ap_coverage, color='orange', alpha=0.7)
        ax4.set_ylabel('Coverage Efficiency (%)')
        ax4.set_title('AP Performance Comparison')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, ap_coverage):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        self.canvas_analysis.draw()
    
    def draw_floor_plan_2d(self, ax):
        """Draw floor plan elements on 2D plot"""
        if not self.floor_plan_data:
            return
        
        for element in self.floor_plan_data:
            if len(element.points) >= 2:
                xs, ys = zip(*element.points)
                
                if element.type == 'wall':
                    ax.plot(xs, ys, 'k-', linewidth=3, alpha=0.8)
                elif element.type == 'room':
                    ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.6)
                elif element.type == 'obstacle':
                    ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.6)
    
    def draw_floor_plan_3d(self, ax):
        """Draw floor plan elements on 3D plot"""
        if not self.floor_plan_data:
            return
        
        for element in self.floor_plan_data:
            if len(element.points) >= 2:
                xs, ys = zip(*element.points)
                zs = [0] * len(xs)  # Floor level
                zs_top = [element.height] * len(xs)  # Ceiling level
                
                if element.type == 'wall':
                    # Draw wall as vertical surfaces
                    for i in range(len(xs) - 1):
                        wall_x = [xs[i], xs[i+1], xs[i+1], xs[i], xs[i]]
                        wall_y = [ys[i], ys[i+1], ys[i+1], ys[i], ys[i]]
                        wall_z = [0, 0, element.height, element.height, 0]
                        ax.plot(wall_x, wall_y, wall_z, 'k-', linewidth=2, alpha=0.8)
    
    def rotate_3d_view(self):
        """Rotate the 3D view"""
        if hasattr(self, 'figure_3d'):
            ax = self.figure_3d.gca()
            if hasattr(ax, 'view_init'):
                current_azim = ax.azim
                ax.view_init(elev=30, azim=current_azim + 30)
                self.canvas_3d.draw()
    
    def generate_coverage_report(self):
        """Generate detailed coverage report"""
        if not self.heatmap_data:
            return
        
        # Calculate statistics
        coverage_data = self.heatmap_data['coverage']
        signal_data = self.heatmap_data['signal_strength']
        sinr_data = self.heatmap_data['sinr']
        
        coverage_percent = np.mean(coverage_data) * 100
        avg_signal = np.mean(signal_data)
        min_signal = np.min(signal_data)
        max_signal = np.max(signal_data)
        avg_sinr = np.mean(sinr_data)
        
        # Create report window
        report_window = tk.Toplevel(self.parent)
        report_window.title("Coverage Analysis Report")
        report_window.geometry("500x400")
        
        # Report text
        report_text = tk.Text(report_window, wrap=tk.WORD, padx=10, pady=10)
        report_text.pack(fill=tk.BOTH, expand=True)
        
        report_content = f"""
WiFi Coverage Analysis Report
============================

Coverage Statistics:
- Overall Coverage: {coverage_percent:.1f}%
- Number of Access Points: {len(self.access_points)}
- Building Area: {(self.building_bounds[2] - self.building_bounds[0]) * (self.building_bounds[3] - self.building_bounds[1]):.1f} m²

Signal Strength Analysis:
- Average Signal Strength: {avg_signal:.1f} dBm
- Minimum Signal Strength: {min_signal:.1f} dBm
- Maximum Signal Strength: {max_signal:.1f} dBm

Quality Metrics:
- Average SINR: {avg_sinr:.1f} dB
- Areas with Good Coverage (>-60 dBm): {np.sum(signal_data > -60) / signal_data.size * 100:.1f}%
- Areas with Fair Coverage (-60 to -70 dBm): {np.sum((signal_data >= -70) & (signal_data <= -60)) / signal_data.size * 100:.1f}%
- Areas with Poor Coverage (<-70 dBm): {np.sum(signal_data < -70) / signal_data.size * 100:.1f}%

Recommendations:
"""
        
        # Add recommendations based on analysis
        if coverage_percent < 85:
            report_content += "- Consider adding more access points to improve coverage\n"
        if avg_signal < -65:
            report_content += "- Overall signal strength is low, consider repositioning APs\n"
        if avg_sinr < 15:
            report_content += "- High interference detected, consider channel optimization\n"
        
        report_content += f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        report_text.insert(tk.END, report_content)
        report_text.config(state=tk.DISABLED)
    
    def export_analysis_data(self):
        """Export analysis data to file"""
        if not self.heatmap_data:
            return
        
        from tkinter import filedialog
        import json
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Prepare data for export (convert numpy arrays to lists)
                export_data = {
                    'coverage_percentage': float(np.mean(self.heatmap_data['coverage']) * 100),
                    'signal_statistics': {
                        'average': float(np.mean(self.heatmap_data['signal_strength'])),
                        'minimum': float(np.min(self.heatmap_data['signal_strength'])),
                        'maximum': float(np.max(self.heatmap_data['signal_strength'])),
                        'std_deviation': float(np.std(self.heatmap_data['signal_strength']))
                    },
                    'sinr_statistics': {
                        'average': float(np.mean(self.heatmap_data['sinr'])),
                        'minimum': float(np.min(self.heatmap_data['sinr'])),
                        'maximum': float(np.max(self.heatmap_data['sinr']))
                    },
                    'access_points': [
                        {
                            'name': ap.name,
                            'x': ap.x,
                            'y': ap.y,
                            'z': ap.z,
                            'power': ap.power,
                            'frequency': ap.frequency
                        }
                        for ap in self.access_points
                    ],
                    'building_bounds': self.building_bounds,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                tk.messagebox.showinfo("Success", "Analysis data exported successfully!")
                
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to export data: {str(e)}")