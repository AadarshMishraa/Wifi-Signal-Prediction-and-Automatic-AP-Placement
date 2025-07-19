# Full-featured BuildingVisualizer with Shapes, Plotting, and Coverage Checks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
from matplotlib import patheffects
from typing import List, Tuple, Dict, Union, Optional
from src.physics.materials import Material, AdvancedMaterial, MATERIALS
import seaborn as sns
import os
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
import cv2
from matplotlib.path import Path as MplPath

class BuildingVisualizer:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.materials_grid: List[List[Union[Material, AdvancedMaterial]]] = [[MATERIALS['air'] for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self._materials_definitions = MATERIALS
        self.walls = []
        self.custom_shapes = []
        self.material_colors = {
            'concrete': '#808080', 'glass': '#ADD8E6', 'wood': '#8B4513', 'drywall': '#F5F5F5',
            'metal': '#C0C0C0', 'brick': "#A52929", 'plaster': '#FFFACD', 'tile': '#D3D3D3',
            'stone': '#A9A9A9', 'asphalt': '#696969', 'carpet': '#B22222', 'plastic': '#FFB6C1',
            'foam': '#F0E68C', 'fabric': '#DDA0DD', 'paper': '#FFF0F5', 'ceramic': '#FAFAD2', 'rubber': '#FF6347'
        }
        # --- NEW: Store all rectangular/circular/polygonal regions for AP placement ---
        self.regions = []  # List of dicts: {'x':..., 'y':..., 'width':..., 'height':..., 'material':...}

    def add_material(self, material: Union[Material, AdvancedMaterial], x: float, y: float, w: float, h: float):
        """
        Add a rectangular region of material to the grid. Accepts both Material and AdvancedMaterial.
        """
        self.walls.append((material, x, y, w, h))
        x1 = int(x / self.resolution)
        y1 = int(y / self.resolution)
        x2 = int((x + w) / self.resolution)
        y2 = int((y + h) / self.resolution)
        for i in range(max(0, y1), min(self.grid_height, y2)):
            for j in range(max(0, x1), min(self.grid_width, x2)):
                self.materials_grid[i][j] = material
        # --- NEW: Record region ---
        self.regions.append({'x': x, 'y': y, 'width': w, 'height': h, 'material': material.name})

    def add_circular_material(self, material: Material, center: tuple, radius: float):
        cx, cy = center
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                x = j * self.resolution
                y = i * self.resolution
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    self.materials_grid[i][j] = material
        self.custom_shapes.append(('circle', material, center, radius))
        # --- NEW: Record circular region as bounding box ---
        self.regions.append({'x': cx - radius, 'y': cy - radius, 'width': 2*radius, 'height': 2*radius, 'material': material.name, 'shape': 'circle', 'center': center, 'radius': radius})

    def add_polygon_material(self, material: Material, vertices: list):
        mpl_poly = MplPolygon(vertices)
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                x = j * self.resolution
                y = i * self.resolution
                if mpl_poly.contains_point((x, y)):
                    self.materials_grid[i][j] = material
        self.custom_shapes.append(('polygon', material, vertices))
        # --- NEW: Record polygon region as bounding box ---
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        self.regions.append({'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y, 'material': material.name, 'shape': 'polygon', 'vertices': vertices})

    def compute_coverage_percentage(self, signal_grid, threshold=-50):
        total_points = signal_grid.size
        covered_points = np.sum(signal_grid >= threshold)
        return (covered_points / total_points) * 100

    def ensure_minimum_coverage(self, signal_grid, threshold=-50, required_percentage=90):
        return self.compute_coverage_percentage(signal_grid, threshold) >= required_percentage

    def plot_signal_strength(self, rssi_grid, points, ap_location, output_path):
        plt.figure(figsize=(14, 8))
        main_ax = plt.gca()
        points = np.array(points)
        x_unique = np.unique(points[:, 0])
        y_unique = np.unique(points[:, 1])
        # Flip the grid vertically to ensure Y increases upwards
        rssi_grid_to_plot = np.flipud(rssi_grid)
        im = plt.imshow(rssi_grid_to_plot,
                        extent=(x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()),
                        origin='lower',
                        cmap='RdYlBu_r',
                        aspect='equal',
                        interpolation='gaussian')
        cbar = plt.colorbar(im, label='Signal Strength (dBm)')
        cbar.ax.tick_params(labelsize=9)

        # === BUILDING REGIONS OVERLAY ===
        # Draw building walls and materials with enhanced visibility
        material_patches = []
        seen_materials = set()
        
        # Draw walls and materials with better styling
        for material, x, y, w, h in self.walls:
            if material.name not in seen_materials:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                alpha=0.7, linewidth=2, label=material.name)
                material_patches.append(patch)
                seen_materials.add(material.name)
            
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                           alpha=0.7, linewidth=2)
            main_ax.add_patch(rect)
            
            # Add material label in the center of each region
            if w > 5 and h > 3:  # Only label larger regions
                main_ax.text(x + w/2, y + h/2, material.name.upper(), 
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Draw custom shapes
        for shape_type, material, *params in self.custom_shapes:
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            if shape_type == 'circle':
                center, radius = params
                circ = Circle(center, radius, facecolor=color, edgecolor='black', 
                            alpha=0.7, linewidth=2)
                main_ax.add_patch(circ)
            elif shape_type == 'polygon':
                vertices = params[0]
                poly = MplPolygon(vertices, closed=True, facecolor=color, 
                                edgecolor='black', alpha=0.7, linewidth=2)
                main_ax.add_patch(poly)

        # === AP LOCATION ===
        if ap_location is not None:
            ap_num = '1'
            import re
            match = re.search(r'coverage_AP(\d+)\.png', output_path)
            if match:
                ap_num = match.group(1)
            
            # Enhanced AP marker
            plt.plot(ap_location[0], ap_location[1], 'r*', markersize=50, zorder=10)
            plt.text(ap_location[0], ap_location[1], ap_num,
                     fontsize=14,
                     color='white',
                     bbox=dict(facecolor='red', edgecolor='black', alpha=1.0, pad=0.5),
                     ha='center', va='center', zorder=11)

            # Add AP name
            plt.text(ap_location[0], ap_location[1] - 3, f'AP{ap_num}',
                     fontsize=12, color='black', weight='bold', ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        # Enhanced legend
        if material_patches:
            plt.legend(handles=material_patches, title='Building Materials', 
                      bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
        
        plt.title('WiFi Signal Strength Map with Building Layout')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    def plot_signal_statistics(self, rssi_by_ap: Dict[str, np.ndarray], plots_dir: str):
        """
        Generate statistical plots for signal strength analysis.
        
        Args:
            rssi_by_ap: Dictionary of AP names to RSSI grids
            plots_dir: Directory to save plots
        """
        if not rssi_by_ap:
            return
            
        # 1. Average Signal Strength Plot
        plt.figure(figsize=(12, 8))
        avg_signals = []
        ap_names = []
        
        for ap_name, rssi_grid in rssi_by_ap.items():
            avg_signals.append(np.mean(rssi_grid))
            ap_names.append(ap_name)
        
        plt.bar(ap_names, avg_signals, color='skyblue', alpha=0.7)
        plt.title('Average Signal Strength by Access Point')
        plt.xlabel('Access Point')
        plt.ylabel('Average Signal Strength (dBm)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'average_signal_strength.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Coverage Area Plot
        plt.figure(figsize=(12, 8))
        coverage_areas = []
        
        for ap_name, rssi_grid in rssi_by_ap.items():
            # Calculate coverage area (points with signal >= -67 dBm)
            covered_points = np.sum(rssi_grid >= -67)
            total_points = rssi_grid.size
            coverage_percentage = (covered_points / total_points) * 100
            coverage_areas.append(coverage_percentage)
        
        plt.bar(ap_names, coverage_areas, color='lightgreen', alpha=0.7)
        plt.title('Coverage Area by Access Point (≥ -67 dBm)')
        plt.xlabel('Access Point')
        plt.ylabel('Coverage Area (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'coverage_area.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Signal Distribution Plot
        plt.figure(figsize=(12, 8))
        
        for ap_name, rssi_grid in rssi_by_ap.items():
            rssi_flat = rssi_grid.flatten()
            plt.hist(rssi_flat, bins=50, alpha=0.6, label=ap_name, density=True)
        
        plt.title('Signal Strength Distribution')
        plt.xlabel('Signal Strength (dBm)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'signal_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined_coverage(self, rssi_grids: List[np.ndarray], ap_locations: dict, output_path: str):
        """
        Create a simplified combined coverage plot showing building layout, 
        AP locations, and coverage contours without heat maps.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
        from matplotlib.lines import Line2D

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # === BUILDING LAYOUT ===
        # Draw building walls and materials with clean styling
        material_patches = []
        seen_materials = set()
        
        # Draw walls and materials
        for material, x, y, w, h in self.walls:
            if material.name not in seen_materials:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                alpha=0.6, linewidth=1, label=material.name)
                material_patches.append(patch)
                seen_materials.add(material.name)
            
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                           alpha=0.6, linewidth=1)
            ax.add_patch(rect)
        
        # Draw custom shapes
        for shape_type, material, *params in self.custom_shapes:
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            if shape_type == 'circle':
                center, radius = params
                circ = Circle(center, radius, facecolor=color, edgecolor='black', 
                            alpha=0.6, linewidth=1)
                ax.add_patch(circ)
            elif shape_type == 'polygon':
                vertices = params[0]
                poly = MplPolygon(vertices, closed=True, facecolor=color, 
                                edgecolor='black', alpha=0.6, linewidth=1)
                ax.add_patch(poly)

        # === AP LOCATIONS ===
        # Plot AP locations with simple, clear markers
        ap_handles = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
                 '#FF8C42', '#8B4513', '#32CD32', '#9370DB', '#20B2AA', '#FF69B4']
        
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            # Handle both 2-tuple (x, y) and 4-tuple (x, y, z, tx_power) coordinates
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
                z = ap_coords[2] if len(ap_coords) > 2 else 0
                tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
            else:
                continue  # Skip invalid coordinates
                
            color = colors[i % len(colors)]
            
            # Plot AP as a simple circle with number
            ap_circle = Circle((x, y), 2.0, facecolor=color, edgecolor='black', 
                             linewidth=2, alpha=0.9, zorder=10)
            ax.add_patch(ap_circle)
            
            # Add AP number and additional info
            label_text = f"{ap_name.replace('AP', '')}\n{z:.1f}m\n{tx_power:.0f}dBm"
            ax.text(x, y, label_text, fontsize=10, color='white', 
                   weight='bold', ha='center', va='center', zorder=11)
            
            # Create legend handle
            h = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, markeredgecolor='black', markeredgewidth=2, 
                      label=f"{ap_name} (z={z:.1f}m, {tx_power:.0f}dBm)")
            ap_handles.append(h)

        # === COVERAGE CONTOURS ===
        # Combine RSSI grids for contour analysis
        combined_grid = np.max(np.stack(rssi_grids), axis=0)
        
        # Add contour lines for coverage levels
        coverage_levels = [-67, -50, -40]  # dBm levels for different coverage quality
        colors_contour = ['red', 'orange', 'green']
        
        for level, color in zip(coverage_levels, colors_contour):
            if np.min(combined_grid) <= level <= np.max(combined_grid):
                contour = ax.contour(combined_grid, levels=[level], colors=color, 
                                   linewidths=2, alpha=0.8, linestyles='--')
                ax.clabel(contour, inline=True, fontsize=8, fmt=f'{level} dBm')

        # === PLOT STYLING ===
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # Add grid with subtle styling
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Labels and title
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('WiFi Coverage Map with Building Layout', fontsize=16, fontweight='bold', pad=20)
        
        # Add legends
        legend1 = None
        if material_patches:
            legend1 = ax.legend(handles=material_patches, title='Building Materials', 
                              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
            ax.add_artist(legend1)
        
        ax.legend(handles=ap_handles, title='Access Points', 
                 bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=10)

        # Clean up layout
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()

    def compare_algorithms_plot(self, ap_results: dict, output_path: str):
        plt.figure(figsize=(10, 6))
        names = list(ap_results.keys())
        values = [ap_results[name] for name in names]

        bars = plt.bar(names, values, color='skyblue')
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.1f}%',
                     ha='center', va='bottom', fontsize=12)

        plt.title('Coverage Comparison by Algorithm')
        plt.ylabel('Coverage ≥ -50 dBm (%)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()

    def get_building_perimeter_polygon(self):
   
        return [(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)]

    def plot_heat_map(self, rssi_grids: List[np.ndarray], ap_locations: dict, output_path: str):
        """
        Create a separate heat map visualization showing signal strength distribution.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
        from matplotlib.lines import Line2D

        # Create figure with subplots for main heat map and coverage scale
        fig = plt.figure(figsize=(18, 12))
        
        # Main heat map plot (left side)
        ax_main = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        
        # Coverage scale plot (right side)
        ax_scale = plt.subplot2grid((1, 4), (0, 3), colspan=1)

        # === HEAT MAP ===
        # Combine RSSI grids and create heat map
        combined_grid = np.max(np.stack(rssi_grids), axis=0)
        smoothed_grid = gaussian_filter(combined_grid, sigma=1.0)
        # Flip the grid vertically to ensure Y increases upwards
        smoothed_grid_to_plot = np.flipud(smoothed_grid)
        im = ax_main.imshow(smoothed_grid_to_plot, origin='lower', cmap='RdYlBu_r', 
                           aspect='equal', alpha=0.8, interpolation='bilinear')
        
        # Add colorbar for signal strength
        cbar = plt.colorbar(im, ax=ax_main, label='Signal Strength (dBm)', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

        # === AP LOCATIONS ===
        # Plot AP locations with simple markers
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
                 '#FF8C42', '#8B4513', '#32CD32', '#9370DB', '#20B2AA', '#FF69B4']
        
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            # Handle both 2-tuple (x, y) and 4-tuple (x, y, z, tx_power) coordinates
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
                z = ap_coords[2] if len(ap_coords) > 2 else 0
                tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
            else:
                continue  # Skip invalid coordinates
                
            color = colors[i % len(colors)]
            
            # Plot AP as a simple circle with number
            ap_circle = Circle((x, y), 2.0, facecolor=color, edgecolor='black', 
                             linewidth=2, alpha=0.9, zorder=10)
            ax_main.add_patch(ap_circle)
            
            # Add AP number and additional info
            label_text = f"{ap_name.replace('AP', '')}\n{z:.1f}m\n{tx_power:.0f}dBm"
            ax_main.text(x, y, label_text, fontsize=10, color='white', 
                        weight='bold', ha='center', va='center', zorder=11)

        # === MAIN PLOT STYLING ===
        ax_main.set_xlim(0, self.width)
        ax_main.set_ylim(0, self.height)
        ax_main.set_aspect('equal')
        
        # Add grid with subtle styling
        ax_main.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Labels and title
        ax_main.set_xlabel('X (meters)', fontsize=12)
        ax_main.set_ylabel('Y (meters)', fontsize=12)
        ax_main.set_title('WiFi Signal Strength Heat Map', fontsize=16, fontweight='bold', pad=20)

        # === COVERAGE SCALE ===
        # Calculate coverage percentages for different signal strength thresholds
        coverage_data = []
        thresholds = [-80, -70, -60, -50, -40, -30]
        threshold_labels = ['Poor\n(-80 dBm)', 'Fair\n(-70 dBm)', 'Good\n(-60 dBm)', 
                           'Very Good\n(-50 dBm)', 'Excellent\n(-40 dBm)', 'Outstanding\n(-30 dBm)']
        
        for threshold in thresholds:
            covered_points = np.sum(combined_grid >= threshold)
            total_points = combined_grid.size
            coverage_percent = (covered_points / total_points) * 100
            coverage_data.append(coverage_percent)
        
        # Create coverage scale bar chart
        bars = ax_scale.barh(range(len(thresholds)), coverage_data, 
                            color=['#FF4444', '#FF8844', '#FFCC44', '#44FF44', '#44CCFF', '#4444FF'],
                            alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for i, (bar, percent) in enumerate(zip(bars, coverage_data)):
            ax_scale.text(percent + 1, bar.get_y() + bar.get_height()/2, 
                         f'{percent:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold')
        
        # Scale styling
        ax_scale.set_yticks(range(len(thresholds)))
        ax_scale.set_yticklabels(threshold_labels, fontsize=10)
        ax_scale.set_xlabel('Coverage Area (%)', fontsize=12)
        ax_scale.set_title('Coverage Analysis', fontsize=14, fontweight='bold', pad=15)
        ax_scale.grid(True, alpha=0.3, axis='x')
        ax_scale.set_xlim(0, 105)  # Give some space for labels
        
        # Add summary statistics
        total_coverage = coverage_data[1]  # Coverage at -70 dBm (fair coverage)
        ax_scale.text(0.5, 0.95, f'Total Coverage: {total_coverage:.1f}%', 
                     transform=ax_scale.transAxes, ha='center', va='top',
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Clean up layout
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()

    def plot_ap_placement_on_floor_plan(self, ap_locations: dict, rssi_grids: Optional[List[np.ndarray]] = None, 
                                       output_path: str = "ap_placement_floor_plan.png", 
                                       show_coverage_areas: bool = True, 
                                       show_signal_heatmap: bool = False):
        """
        Overlay AP placement on the building layout with coverage visualization.
        AP circles are small, red, numbered, with no overlap or extra info.
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        # Draw building layout (walls/materials)
        material_patches = []
        seen_materials = set()
        for material, x, y, w, h in self.walls:
            if material.name not in seen_materials:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                alpha=0.6, linewidth=1, label=material.name)
                material_patches.append(patch)
                seen_materials.add(material.name)
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                           alpha=0.6, linewidth=1)
            ax.add_patch(rect)
        # Draw custom shapes
        for shape_type, material, *params in self.custom_shapes:
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            if shape_type == 'circle':
                center, radius = params
                circ = Circle(center, radius, facecolor=color, edgecolor='black', 
                            alpha=0.6, linewidth=1)
                ax.add_patch(circ)
            elif shape_type == 'polygon':
                vertices = params[0]
                poly = MplPolygon(vertices, closed=True, facecolor=color, 
                                edgecolor='black', alpha=0.6, linewidth=1)
                ax.add_patch(poly)
        # Plot APs as small red circles with numbers
        ap_handles = []
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
            else:
                continue
            # Small red circle
            ap_circle = Circle((x, y), 0.7, facecolor='red', edgecolor='black', linewidth=2, alpha=0.95, zorder=10)
            ax.add_patch(ap_circle)
            # White AP number inside
            ap_num = int(ap_name.replace('AP', '')) if ap_name.startswith('AP') else i+1
            ax.text(x, y, str(ap_num), fontsize=10, color='white', weight='bold', ha='center', va='center', zorder=11)
            # Legend handle
            from matplotlib.lines import Line2D
            h = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', markeredgewidth=2, label=f"AP{ap_num}")
            ap_handles.append(h)
            # Coverage circle (smaller, lighter, dashed red)
            if show_coverage_areas:
                coverage_radius = 10.0  # meters, smaller for clarity
                coverage_circle = Circle((x, y), coverage_radius, facecolor='none', alpha=0.15, edgecolor='red', linewidth=1.5, linestyle='--', zorder=5)
                ax.add_patch(coverage_circle)
        # Heatmap overlay (combined)
        if show_signal_heatmap and rssi_grids:
            combined_grid = np.max(np.stack(rssi_grids), axis=0)
            im = ax.imshow(combined_grid, origin='lower', cmap='Reds', alpha=0.4, extent=(0, self.width, 0, self.height), zorder=3, interpolation='bilinear')
            cbar = plt.colorbar(im, ax=ax, label='Signal Strength (dBm)', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
        # Plot styling
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Title
        title = "WiFi Access Point Placement on Building Layout (Red Circles = APs)"
        if show_signal_heatmap:
            title += " (with combined heatmap)"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        # Legend outside
        if ap_handles:
            ax.legend(handles=ap_handles, title='APs', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"AP placement on building layout saved to: {output_path}")
        return True

    def plot_per_ap_heatmaps(self, rssi_grids: List[np.ndarray], ap_locations: dict, output_dir: str):
        """
        Plot a heatmap for each AP, showing only that AP as a red circle with its number.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            fig, ax = plt.subplots(figsize=(12, 10))
            # Heatmap for this AP
            grid = rssi_grids[i]
            im = ax.imshow(grid, origin='lower', cmap='Reds', alpha=0.7, extent=(0, self.width, 0, self.height), zorder=3, interpolation='bilinear')
            cbar = plt.colorbar(im, ax=ax, label='Signal Strength (dBm)', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            # AP as red circle with number
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
                ap_circle = Circle((x, y), 0.7, facecolor='red', edgecolor='black', linewidth=2, alpha=0.95, zorder=10)
                ax.add_patch(ap_circle)
                ap_num = int(ap_name.replace('AP', '')) if ap_name.startswith('AP') else i+1
                ax.text(x, y, str(ap_num), fontsize=10, color='white', weight='bold', ha='center', va='center', zorder=11)
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"AP{ap_num} Coverage Heatmap", fontsize=14, fontweight='bold')
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"ap{ap_num}_heatmap.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            print(f"Per-AP heatmap saved to: {out_path}")

    def plot_combined_heatmap(self, rssi_grids: List[np.ndarray], ap_locations: dict, output_path: str):
        """
        Plot a combined heatmap (max signal at each point) with all APs as red circles with numbers.
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        combined_grid = np.max(np.stack(rssi_grids), axis=0)
        im = ax.imshow(combined_grid, origin='lower', cmap='Reds', alpha=0.7, extent=(0, self.width, 0, self.height), zorder=3, interpolation='bilinear')
        cbar = plt.colorbar(im, ax=ax, label='Signal Strength (dBm)', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        # Plot APs as red circles with numbers
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
                ap_circle = Circle((x, y), 0.7, facecolor='red', edgecolor='black', linewidth=2, alpha=0.95, zorder=10)
                ax.add_patch(ap_circle)
                ap_num = int(ap_name.replace('AP', '')) if ap_name.startswith('AP') else i+1
                ax.text(x, y, str(ap_num), fontsize=10, color='white', weight='bold', ha='center', va='center', zorder=11)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Combined Coverage Heatmap (Max Signal)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Combined heatmap saved to: {output_path}")

    def plot_coverage_with_floor_plan_regions(self, rssi_grids: List[np.ndarray], ap_locations: dict, 
                                            output_path: str, show_heatmap: bool = True):
        """
        Create a comprehensive coverage plot showing signal strength, building regions, and AP locations.
        
        Args:
            rssi_grids: List of RSSI grids for each AP
            ap_locations: Dictionary of AP names to (x, y) coordinates
            output_path: Path to save the output image
            show_heatmap: Whether to show signal strength heatmap
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
        from matplotlib.lines import Line2D

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # === SIGNAL STRENGTH HEATMAP ===
        if show_heatmap and rssi_grids:
            # Combine RSSI grids for overall coverage
            combined_grid = np.max(np.stack(rssi_grids), axis=0)
            
            # Create heatmap with transparency
            im = ax.imshow(combined_grid, origin='lower', cmap='RdYlBu_r', 
                          alpha=0.6, extent=(0, self.width, 0, self.height), 
                          zorder=1, interpolation='bilinear')
            
            # Add colorbar for signal strength
            cbar = plt.colorbar(im, ax=ax, label='Signal Strength (dBm)', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

        # === BUILDING REGIONS OVERLAY ===
        # Draw building walls and materials with enhanced visibility
        material_patches = []
        seen_materials = set()
        
        # Draw walls and materials
        for material, x, y, w, h in self.walls:
            if material.name not in seen_materials:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                alpha=0.8, linewidth=2, label=material.name)
                material_patches.append(patch)
                seen_materials.add(material.name)
            
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                           alpha=0.8, linewidth=2, zorder=5)
            ax.add_patch(rect)
            
            # Add material label in the center of each region
            if w > 5 and h > 3:  # Only label larger regions
                ax.text(x + w/2, y + h/2, material.name.upper(), 
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                       zorder=6)
        
        # Draw custom shapes
        for shape_type, material, *params in self.custom_shapes:
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            if shape_type == 'circle':
                center, radius = params
                circ = Circle(center, radius, facecolor=color, edgecolor='black', 
                            alpha=0.8, linewidth=2, zorder=5)
                ax.add_patch(circ)
            elif shape_type == 'polygon':
                vertices = params[0]
                poly = MplPolygon(vertices, closed=True, facecolor=color, 
                                edgecolor='black', alpha=0.8, linewidth=2, zorder=5)
                ax.add_patch(poly)

        # === AP LOCATIONS ===
        ap_handles = []
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080']
        
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            # Handle both 2-tuple (x, y) and 4-tuple (x, y, z, tx_power) coordinates
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
                z = ap_coords[2] if len(ap_coords) > 2 else 0
                tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
            else:
                continue  # Skip invalid coordinates
                
            color = colors[i % len(colors)]
            
            # Plot AP as a prominent circle with number
            ap_circle = Circle((x, y), 1.5, facecolor=color, edgecolor='black', 
                             linewidth=3, alpha=0.9, zorder=10)
            ax.add_patch(ap_circle)
            
            # Add AP number inside circle with additional info
            label_text = f"{ap_name.replace('AP', '')}\n{z:.1f}m\n{tx_power:.0f}dBm"
            ax.text(x, y, label_text, fontsize=12, color='white', 
                   weight='bold', ha='center', va='center', zorder=11)
            
            # Add AP name below circle
            ax.text(x, y - 2.5, ap_name, fontsize=11, color='black', 
                   weight='bold', ha='center', va='center', zorder=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            # Create legend handle
            h = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=12, markeredgecolor='black', markeredgewidth=2, 
                      label=f"{ap_name} (z={z:.1f}m, {tx_power:.0f}dBm)")
            ap_handles.append(h)

        # === COVERAGE CONTOURS ===
        if rssi_grids:
            combined_grid = np.max(np.stack(rssi_grids), axis=0)
            
            # Add contour lines for different coverage levels
            coverage_levels = [-67, -50, -40]  # dBm levels
            contour_colors = ['red', 'orange', 'green']
            
            for level, color in zip(coverage_levels, contour_colors):
                if np.min(combined_grid) <= level <= np.max(combined_grid):
                    contour = ax.contour(combined_grid, levels=[level], colors=color, 
                                       linewidths=2, alpha=0.9, linestyles='--', zorder=4)
                    ax.clabel(contour, inline=True, fontsize=9, fmt=f'{level} dBm')

        # === PLOT STYLING ===
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # Add grid for reference
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        title = "WiFi Coverage with Building Layout"
        if show_heatmap:
            title += " (Signal Strength Heatmap)"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add legends
        if material_patches:
            legend1 = ax.legend(handles=material_patches, title='Building Materials', 
                              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            ax.add_artist(legend1)
        
        if ap_handles:
            ax.legend(handles=ap_handles, title='Access Points', 
                     bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=10)
        
        # Add building information
        info_text = f"Building: {self.width}m × {self.height}m\nAPs: {len(ap_locations)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor="white", alpha=0.8))
        
        # Clean up layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Coverage plot with floor plan regions saved to: {output_path}")
        return True

    def plot_coverage_on_floor_plan_image(self, rssi_grids: List[np.ndarray], ap_locations: dict, output_path: str, show_regions: bool = True):
        """
        Plot the coverage heatmap and APs on the building layout with programmatic overlays.
        Only plot within the building perimeter polygon if available.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
        from matplotlib.lines import Line2D
        import numpy as np
        from matplotlib.path import Path as MplPath
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # === BUILDING LAYOUT OVERLAY ===
        # Draw building walls and materials with enhanced visibility
        material_patches = []
        seen_materials = set()
        
        # Draw walls and materials
        for material, x, y, w, h in self.walls:
            if material.name not in seen_materials:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                alpha=0.6, linewidth=1, label=material.name)
                material_patches.append(patch)
                seen_materials.add(material.name)
            
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                           alpha=0.6, linewidth=1)
            ax.add_patch(rect)
        
        # Draw custom shapes
        for shape_type, material, *params in self.custom_shapes:
            color = self.material_colors.get(material.name.lower(), '#FFFFFF')
            if shape_type == 'circle':
                center, radius = params
                circ = Circle(center, radius, facecolor=color, edgecolor='black', 
                            alpha=0.6, linewidth=1)
                ax.add_patch(circ)
            elif shape_type == 'polygon':
                vertices = params[0]
                poly = MplPolygon(vertices, closed=True, facecolor=color, 
                                edgecolor='black', alpha=0.6, linewidth=1)
                ax.add_patch(poly)
        
        # Get building perimeter polygon if available
        polygon = None
        if hasattr(self, 'get_building_perimeter_polygon'):
            polygon = self.get_building_perimeter_polygon()
        
        # Plot the coverage heatmap, masking to the polygon if available
        if rssi_grids:
            combined_grid = np.max(np.stack(rssi_grids), axis=0)
            y_grid, x_grid = combined_grid.shape
            x = np.linspace(0, self.width, x_grid)
            y = np.linspace(0, self.height, y_grid)
            X, Y = np.meshgrid(x, y)
            # Flip the grid vertically to ensure Y increases upwards
            combined_grid_to_plot = np.flipud(combined_grid)
            if polygon:
                path = MplPath(polygon)
                mask = np.array([path.contains_point((x, y)) for x, y in zip(X.flatten(), Y.flatten())])
                mask = mask.reshape(X.shape)
                masked_grid = np.ma.masked_where(~mask, combined_grid_to_plot)
                im = ax.imshow(masked_grid, origin='lower', cmap='RdYlBu_r',
                              alpha=0.5, extent=(0, self.width, 0, self.height),
                              zorder=2, interpolation='bilinear')
            else:
                im = ax.imshow(combined_grid_to_plot, origin='lower', cmap='RdYlBu_r',
                              alpha=0.5, extent=(0, self.width, 0, self.height),
                              zorder=2, interpolation='bilinear')
            cbar = plt.colorbar(im, ax=ax, label='Signal Strength (dBm)', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
        
        # Optionally overlay regions
        if show_regions:
            material_patches = []
            seen_materials = set()
            for material, x, y, w, h in self.walls:
                if material.name not in seen_materials:
                    color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                    patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                    alpha=0.7, linewidth=2, label=material.name)
                    material_patches.append(patch)
                    seen_materials.add(material.name)
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                               alpha=0.3, linewidth=2, zorder=3)
                ax.add_patch(rect)
                if w > 5 and h > 3:
                    ax.text(x + w/2, y + h/2, material.name.upper(),
                            ha='center', va='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7), zorder=4)
            # Custom shapes
            for shape_type, material, *params in self.custom_shapes:
                color = self.material_colors.get(material.name.lower(), '#FFFFFF')
                if shape_type == 'circle':
                    center, radius = params
                    circ = Circle(center, radius, facecolor=color, edgecolor='black', 
                                alpha=0.3, linewidth=2, zorder=3)
                    ax.add_patch(circ)
                elif shape_type == 'polygon':
                    vertices = params[0]
                    poly = MplPolygon(vertices, closed=True, facecolor=color, 
                                    edgecolor='black', alpha=0.3, linewidth=2, zorder=3)
                    ax.add_patch(poly)
        
        # Plot AP locations, masking to polygon if available
        ap_handles = []
        ap_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                    '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080']
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            # Handle both 2-tuple (x, y) and 4-tuple (x, y, z, tx_power) coordinates
            if len(ap_coords) >= 2:
                x, y = ap_coords[0], ap_coords[1]
                z = ap_coords[2] if len(ap_coords) > 2 else 0
                tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
            else:
                continue  # Skip invalid coordinates
                
            if polygon:
                path = MplPath(polygon)
                if not path.contains_point((x, y)):
                    continue  # Skip APs outside the building
            color = ap_colors[i % len(ap_colors)]
            ap_circle = Circle((x, y), 1.5, facecolor=color, edgecolor='black',
                              linewidth=3, alpha=0.9, zorder=10)
            ax.add_patch(ap_circle)
            
            # Add AP number with additional info
            label_text = f"{ap_name.replace('AP', '')}\n{z:.1f}m\n{tx_power:.0f}dBm"
            ax.text(x, y, label_text, fontsize=12, color='white',
                    weight='bold', ha='center', va='center', zorder=11)
            ax.text(x, y - 2.5, ap_name, fontsize=11, color='black',
                    weight='bold', ha='center', va='center', zorder=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            h = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                        markersize=12, markeredgecolor='black', markeredgewidth=2, 
                        label=f"{ap_name} (z={z:.1f}m, {tx_power:.0f}dBm)")
            ap_handles.append(h)
        
        # Plot styling
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('WiFi Coverage and AP Placement on Building Layout', fontsize=16, fontweight='bold', pad=20)
        if ap_handles:
            ax.legend(handles=ap_handles, title='Access Points', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Coverage and AP placement on building layout saved to: {output_path}")
        return True

    def plot_3d_coverage(self, ap_locations_3d, receiver_points_3d, rssi_values=None, materials_grid_3d=None, output_path="coverage_3d.png", z_slice=None):
        """
        Plot a 3D visualization of APs, receivers, and optionally material blocks.
        Args:
            ap_locations_3d: dict of AP name to (x, y, z)
            receiver_points_3d: list of (x, y, z) tuples
            rssi_values: list or array of RSSI values for each receiver (optional, for coloring)
            materials_grid_3d: 3D grid [z][y][x] of materials or stacks (optional)
            output_path: where to save the plot
            z_slice: if set, only plot this z index (for slice visualization)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        
        # Plot APs
        for ap_name, (x, y, z) in ap_locations_3d.items():
            ax.scatter(x, y, z, c='red', marker='*', s=200, label=ap_name)  # type: ignore
            ax.text(x, y, z+0.5, ap_name, color='red', fontsize=12, weight='bold')  # type: ignore
        
        # Plot receivers
        if receiver_points_3d:
            xs, ys, zs = zip(*receiver_points_3d)
            if rssi_values is not None:
                p = ax.scatter(xs, ys, zs, c=rssi_values, cmap='RdYlBu_r', marker='o', s=20, alpha=0.7)  # type: ignore
                fig.colorbar(p, ax=ax, shrink=0.5, label='Signal Strength (dBm)')
            else:
                ax.scatter(xs, ys, zs, c='blue', marker='o', s=10, alpha=0.5)  # type: ignore
        
        # Plot 3D material grid as voxels or cuboids (optional)
        if materials_grid_3d is not None:
            nz = len(materials_grid_3d)
            ny = len(materials_grid_3d[0])
            nx = len(materials_grid_3d[0][0])
            res = self.resolution if hasattr(self, 'resolution') else 0.2
            for z in range(nz):
                if z_slice is not None and z != z_slice:
                    continue
                for y in range(ny):
                    for x in range(nx):
                        stack = materials_grid_3d[z][y][x]
                        if stack and stack[0].name.lower() != 'air':
                            mat = stack[0]
                            color = self.material_colors.get(mat.name.lower(), '#888888')
                            # Use scatter3d instead of bar3d for compatibility
                            ax.scatter(x*res, y*res, z*res, c=color, s=50, alpha=0.2)  # type: ignore
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')  # type: ignore
        ax.set_title('3D WiFi Coverage and AP Placement')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _draw_regions_overlay(self, ax):
        """Draw all regions (rect, circle, polygon) on the given axis."""
        from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
        for region in getattr(self, 'regions', []):
            color = self.material_colors.get(region.get('material', '').lower(), '#DDDDDD')
            shape = region.get('shape', 'rect')
            if shape == 'circle':
                center = region.get('center', (region['x'] + region['width']/2, region['y'] + region['height']/2))
                radius = region.get('radius', region['width']/2)
                circ = Circle(center, radius, facecolor=color, edgecolor='green', alpha=0.25, linewidth=2, zorder=3, linestyle='--')
                ax.add_patch(circ)
            elif shape == 'polygon':
                vertices = region.get('vertices', [])
                if vertices:
                    poly = MplPolygon(vertices, closed=True, facecolor=color, edgecolor='green', alpha=0.25, linewidth=2, zorder=3, linestyle='--')
                    ax.add_patch(poly)
            else:
                rect = Rectangle((region['x'], region['y']), region['width'], region['height'], facecolor=color, edgecolor='green', alpha=0.25, linewidth=2, zorder=3, linestyle='--')
                ax.add_patch(rect)
            # Add material label in the center
            cx = region['x'] + region['width']/2
            cy = region['y'] + region['height']/2
            if region['width'] > 5 and region['height'] > 3:
                ax.text(cx, cy, region.get('material', '').upper(), ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7), zorder=4)

    def _draw_aps_overlay(self, ax, ap_locations: Dict[str, Tuple[float, float]]):
        """Helper to draw AP locations on a given matplotlib axis."""
        if not ap_locations:
            print("Warning: No AP locations provided to draw.")
            return

        for ap_name, coords in ap_locations.items():
            if len(coords) >= 2:
                x, y = coords[0], coords[1]
                ax.plot(x, y, 'X', color='red', markersize=10, markeredgewidth=1.5, label=ap_name)
                # Add text label slightly offset from the marker
                ax.text(x + 0.2, y + 0.2, ap_name, color='red', fontsize=9, ha='left', va='bottom')

    def _plot_hybrid_statistics(self, physics_pred: np.ndarray, hybrid_pred: np.ndarray, plots_dir: str):
        """
        Generates statistical comparison plots between physics and hybrid model predictions.

        Args:
            physics_pred: 2D array of physics model predictions.
            hybrid_pred: 2D array of hybrid model predictions.
            plots_dir: Directory to save plots.
        """
        print("Generating hybrid statistical comparison plots...")

        # Flatten the 2D arrays for statistical analysis
        physics_flat = physics_pred.flatten()
        hybrid_flat = hybrid_pred.flatten()

        # Create a DataFrame for easier plotting
        import pandas as pd
        df_compare = pd.DataFrame({
            'Physics_RSSI': physics_flat,
            'Hybrid_RSSI': hybrid_flat
        })

        # --- Plot 1: Histograms of Signal Strength Distributions ---
        plt.figure(figsize=(10, 6))
        df_compare['Physics_RSSI'].hist(bins=50, alpha=0.7, label='Physics Model', color='blue')
        df_compare['Hybrid_RSSI'].hist(bins=50, alpha=0.7, label='Hybrid Model', color='green')
        plt.title('Distribution of Predicted Signal Strengths')
        plt.xlabel('Signal Strength (dBm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'signal_distribution_comparison.png'))
        plt.close()

        # --- Plot 2: Box Plots for Central Tendency and Spread ---
        plt.figure(figsize=(8, 6))
        plt.boxplot([df_compare['Physics_RSSI'], df_compare['Hybrid_RSSI']],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
        plt.title('Signal Strength Box Plot Comparison')
        plt.ylabel('Signal Strength (dBm)')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(ticks=[1,2], labels=['Physics Model', 'Hybrid Model'])
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'signal_boxplot_comparison.png'))
        plt.close()

        # --- Plot 3: Scatter plot of Physics vs Hybrid ---
        # Only plot scatter if the number of points is manageable to avoid very slow rendering
        if len(physics_flat) < 100000:  # Adjust threshold as needed
            plt.figure(figsize=(8, 8))
            plt.scatter(physics_flat, hybrid_flat, alpha=0.3, s=5, c='purple')
            min_val = min(physics_flat.min(), hybrid_flat.min())
            max_val = max(physics_flat.max(), hybrid_flat.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction (y=x)')
            plt.xlabel('Physics Model Prediction (dBm)')
            plt.ylabel('Hybrid Model Prediction (dBm)')
            plt.title('Physics vs Hybrid Model Predictions')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'physics_vs_hybrid_scatter.png'))
            plt.close()
        else:
            print(f"Skipping physics vs hybrid scatter plot due to high number of points ({len(physics_flat)}).")
        
        print("Hybrid statistical comparison plots generated successfully.")

    def plot_hybrid_comparison(self, physics_pred: np.ndarray, hybrid_pred: np.ndarray,
                             points: List[Tuple[float, float]], ap_locations: Dict[str, Tuple[float, float]], plots_dir: str):
        """
        Create comprehensive comparison plots between physics and hybrid predictions.
        This includes heatmaps and a difference plot, with material and AP overlays.

        Args:
            physics_pred: 2D array of physics model predictions.
            hybrid_pred: 2D array of hybrid model predictions.
            points: List of (x, y) coordinates for precise plot extent.
            ap_locations: Dictionary of AP names and their (x, y) coordinates for overlay.
            plots_dir: Directory to save output plots.
        """
        print("Generating comprehensive hybrid comparison plots...")
        try:
            # Ensure output directory exists
            os.makedirs(plots_dir, exist_ok=True)

            # Determine extent from points
            x_unique = np.unique([p[0] for p in points])
            y_unique = np.unique([p[1] for p in points])
            extent = (float(x_unique.min()), float(x_unique.max()), float(y_unique.min()), float(y_unique.max()))

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))  # Adjusted figsize for better viewing

            # Define common vmin/vmax for RSSI plots
            # These values should be chosen based on typical RSSI ranges for your application
            vmin_rssi, vmax_rssi = -90, -40  # Example: -90 dBm (very weak) to -40 dBm (strong)

            # Plot 1: Physics Model Predictions
            im1 = ax1.imshow(physics_pred, extent=extent, origin='lower',
                             cmap='RdYlBu_r', vmin=vmin_rssi, vmax=vmax_rssi)
            ax1.set_title('Physics Model Predictions')
            ax1.set_xlabel('X (meters)')
            ax1.set_ylabel('Y (meters)')
            fig.colorbar(im1, ax=ax1, shrink=0.8, label='Signal Strength (dBm)')
            self._draw_regions_overlay(ax1)
            self._draw_aps_overlay(ax1, ap_locations)
            ax1.grid(True, alpha=0.2)

            # Plot 2: Hybrid Model Predictions
            im2 = ax2.imshow(hybrid_pred, extent=extent, origin='lower',
                             cmap='RdYlBu_r', vmin=vmin_rssi, vmax=vmax_rssi)
            ax2.set_title('Hybrid Model Predictions')
            ax2.set_xlabel('X (meters)')
            ax2.set_ylabel('Y (meters)')
            fig.colorbar(im2, ax=ax2, shrink=0.8, label='Signal Strength (dBm)')
            self._draw_regions_overlay(ax2)
            self._draw_aps_overlay(ax2, ap_locations)
            ax2.grid(True, alpha=0.2)

            # Plot 3: Difference (Hybrid - Physics)
            diff = hybrid_pred - physics_pred
            # Define vmin/vmax for the difference plot, typically centered around 0
            vmin_diff, vmax_diff = -15, 15  # Example: Difference range of +/- 15 dBm
            im3 = ax3.imshow(diff, extent=extent, origin='lower',
                             cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff)
            ax3.set_title('Difference (Hybrid - Physics) dBm')
            ax3.set_xlabel('X (meters)')
            ax3.set_ylabel('Y (meters)')
            fig.colorbar(im3, ax=ax3, shrink=0.8, label='Difference (dBm)')
            self._draw_regions_overlay(ax3)
            self._draw_aps_overlay(ax3, ap_locations)
            ax3.grid(True, alpha=0.2)

            plt.tight_layout()
            output_filepath = os.path.join(plots_dir, 'hybrid_comparison_heatmaps.png')
            plt.savefig(output_filepath, dpi=300, bbox_inches='tight')  # Using 300 DPI for good balance
            plt.close(fig)  # Close the figure to free memory
            print(f"Comprehensive hybrid comparison heatmaps saved to {output_filepath}")

            # Additional statistical comparison plot
            self._plot_hybrid_statistics(physics_pred, hybrid_pred, plots_dir)

        except ValueError as ve:
            print(f"Input data error in plot_hybrid_comparison: {str(ve)}")
        except Exception as e:
            print(f"An unexpected error occurred in plot_hybrid_comparison: {str(e)}")

    def plot_signal_strength_enhanced(self, rssi_grid: np.ndarray, points: List[Tuple[float, float]],
                                     ap_loc: Union[Tuple[float, float], Dict[str, Tuple[float, float]]], plots_dir: str,
                                     filename: str = 'signal_strength_heatmap.png'):
        """
        Enhanced version of signal strength plotting with material and AP overlays.
        Args:
            rssi_grid: 2D array of RSSI values.
            points: List of (x,y) coordinates corresponding to the grid.
            ap_loc: Location of a single AP (x,y) or a dictionary of APs for combined plot.
            plots_dir: Directory to save the plot.
            filename: Name of the file to save the plot.
        """
        print(f"Generating enhanced signal strength heatmap: {filename}...")
        try:
            os.makedirs(plots_dir, exist_ok=True)

            x_unique = np.unique([p[0] for p in points])
            y_unique = np.unique([p[1] for p in points])
            extent = (float(x_unique.min()), float(x_unique.max()), float(y_unique.min()), float(y_unique.max()))

            fig, ax = plt.subplots(figsize=(12, 8))

            vmin_rssi, vmax_rssi = -90, -40

            im = ax.imshow(rssi_grid, origin='lower', extent=extent, cmap='RdYlBu_r', vmin=vmin_rssi, vmax=vmax_rssi)

            self._draw_regions_overlay(ax)

            # Handle single AP vs multiple APs for drawing
            if isinstance(ap_loc, dict):
                self._draw_aps_overlay(ax, ap_loc)
            else:
                self._draw_aps_overlay(ax, {'AP': ap_loc})  # For individual AP plots, label it generically

            ax.set_title('WiFi Signal Strength Heatmap')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            fig.colorbar(im, ax=ax, label='Signal Strength (dBm)')
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            output_filepath = os.path.join(plots_dir, filename)
            plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Enhanced signal strength heatmap saved to {output_filepath}")

        except Exception as e:
            print(f"Error in plot_signal_strength_enhanced: {str(e)}")

    def plot_model_comparison(self, model_predictions: Dict[str, np.ndarray], 
                             points: List[Tuple[float, float]], 
                             ap_locations: Dict[str, Tuple[float, float]], 
                             plots_dir: str):
        """
        Create comparison plots for different ML models (KNN, Random Forest, CNN, etc.).
        
        Args:
            model_predictions: Dictionary with model names as keys and 2D prediction arrays as values
            points: List of (x, y) coordinates for plot extent
            ap_locations: Dictionary of AP names and coordinates
            plots_dir: Directory to save plots
        """
        print("Generating model comparison plots...")
        try:
            os.makedirs(plots_dir, exist_ok=True)
            
            if not model_predictions:
                print("No model predictions provided for comparison.")
                return
                
            # Determine extent from points
            x_unique = np.unique([p[0] for p in points])
            y_unique = np.unique([p[1] for p in points])
            extent = (float(x_unique.min()), float(x_unique.max()), float(y_unique.min()), float(y_unique.max()))
            
            # Create subplots for each model
            n_models = len(model_predictions)
            fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(6 * ((n_models + 1) // 2), 10))
            if n_models == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Define common color range
            vmin_rssi, vmax_rssi = -90, -40
            
            for idx, (model_name, predictions) in enumerate(model_predictions.items()):
                if idx < len(axes):
                    ax = axes[idx]
                    try:
                        # Ensure predictions is a 2D numpy array
                        if not isinstance(predictions, np.ndarray):
                            predictions = np.array(predictions)
                        
                        # Debug info
                        print(f"Model {model_name}: predictions shape = {predictions.shape}, ndim = {predictions.ndim}")
                        print(f"x_unique: {len(x_unique)}, y_unique: {len(y_unique)}")
                        
                        if predictions.ndim == 1:
                            # Check if we can reshape
                            expected_size = len(y_unique) * len(x_unique)
                            if len(predictions) != expected_size:
                                print(f"Warning: predictions length ({len(predictions)}) != expected size ({expected_size})")
                                # Pad or truncate to match expected size
                                if len(predictions) < expected_size:
                                    # Pad with the last value
                                    predictions = np.pad(predictions, (0, expected_size - len(predictions)), 
                                                       mode='edge')
                                else:
                                    # Truncate
                                    predictions = predictions[:expected_size]
                            
                            # Reshape 1D array to 2D based on points
                            predictions = predictions.reshape(len(y_unique), len(x_unique))
                        
                        # Ensure we have a valid 2D array
                        if predictions.ndim != 2:
                            print(f"Error: predictions is not 2D after processing: {predictions.shape}")
                            continue
                        
                        im = ax.imshow(predictions, extent=extent, origin='lower',
                                      cmap='RdYlBu_r', vmin=vmin_rssi, vmax=vmax_rssi)
                    except Exception as e:
                        print(f"Error processing model {model_name}: {e}")
                        print(f"predictions type: {type(predictions)}")
                        print(f"predictions shape: {getattr(predictions, 'shape', 'no shape')}")
                        continue
                    ax.set_title(f'{model_name} Predictions')
                    ax.set_xlabel('X (meters)')
                    ax.set_ylabel('Y (meters)')
                    self._draw_regions_overlay(ax)
                    self._draw_aps_overlay(ax, ap_locations)
                    ax.grid(True, alpha=0.2)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Signal Strength (dBm)')
            
            # Hide unused subplots
            for idx in range(n_models, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            output_filepath = os.path.join(plots_dir, 'model_comparison_heatmaps.png')
            plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Model comparison heatmaps saved to {output_filepath}")
            
            # Generate statistical comparison
            self._plot_model_statistics(model_predictions, plots_dir)
            
        except Exception as e:
            print(f"Error in plot_model_comparison: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    def _plot_model_statistics(self, model_predictions: Dict[str, np.ndarray], plots_dir: str):
        """
        Generate statistical comparison plots for different models.
        
        Args:
            model_predictions: Dictionary with model names and prediction arrays
            plots_dir: Directory to save plots
        """
        print("Generating model statistical comparison plots...")
        
        # Flatten predictions for statistical analysis
        model_data = {}
        for model_name, predictions in model_predictions.items():
            model_data[model_name] = predictions.flatten()
        
        # Create box plot comparison
        plt.figure(figsize=(10, 6))
        data_to_plot = list(model_data.values())
        labels = list(model_data.keys())
        
        plt.boxplot(data_to_plot, patch_artist=True)
        plt.title('Model Performance Comparison')
        plt.ylabel('Signal Strength (dBm)')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(ticks=range(1, len(labels)+1), labels=labels, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_performance_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create histogram comparison
        plt.figure(figsize=(12, 8))
        for model_name, data in model_data.items():
            plt.hist(data, bins=50, alpha=0.6, label=model_name)
        
        plt.title('Signal Strength Distribution by Model')
        plt.xlabel('Signal Strength (dBm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model statistical comparison plots generated successfully.")

    def plot_individual_ap_coverage(self, rssi_grids: List[np.ndarray], 
                                   ap_locations: Dict[str, Tuple[float, float]], 
                                   points: List[Tuple[float, float]], 
                                   plots_dir: str):
        """
        Generate individual coverage plots for each AP.
        
        Args:
            rssi_grids: List of 2D RSSI grids for each AP
            ap_locations: Dictionary of AP names and coordinates
            points: List of (x, y) coordinates for plot extent
            plots_dir: Directory to save plots
        """
        print("Generating individual AP coverage plots...")
        try:
            os.makedirs(plots_dir, exist_ok=True)
            
            # Determine extent from points
            x_unique = np.unique([p[0] for p in points])
            y_unique = np.unique([p[1] for p in points])
            extent = (float(x_unique.min()), float(x_unique.max()), float(y_unique.min()), float(y_unique.max()))
            
            # Define common color range
            vmin_rssi, vmax_rssi = -90, -40
            
            for idx, (ap_name, ap_coords) in enumerate(ap_locations.items()):
                if idx < len(rssi_grids):
                    rssi_grid = rssi_grids[idx]
                    
                    plt.figure(figsize=(10, 8))
                    im = plt.imshow(rssi_grid, extent=extent, origin='lower',
                                   cmap='RdYlBu_r', vmin=vmin_rssi, vmax=vmax_rssi)
                    
                    plt.title(f'Coverage Heatmap for {ap_name}')
                    plt.xlabel('X (meters)')
                    plt.ylabel('Y (meters)')
                    
                    # Draw materials overlay
                    self._draw_regions_overlay(plt.gca())
                    
                    # Draw AP location
                    if len(ap_coords) >= 2:
                        plt.plot(ap_coords[0], ap_coords[1], 'X', color='red', markersize=15, markeredgewidth=2)
                        plt.text(ap_coords[0] + 0.2, ap_coords[1] + 0.2, ap_name, 
                                color='red', fontsize=12, fontweight='bold')
                    
                    plt.colorbar(im, label='Signal Strength (dBm)')
                    plt.grid(True, alpha=0.2)
                    plt.tight_layout()
                    
                    output_filepath = os.path.join(plots_dir, f'coverage_{ap_name}.png')
                    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Coverage plot for {ap_name} saved to {output_filepath}")
            
            print("Individual AP coverage plots generated successfully.")
            
        except Exception as e:
            print(f"Error in plot_individual_ap_coverage: {str(e)}")

    def plot_combined_coverage_enhanced(self, rssi_grids: List[np.ndarray], 
                                       ap_locations: Dict[str, Tuple[float, float]], 
                                       points: List[Tuple[float, float]], 
                                       plots_dir: str):
        """
        Enhanced combined coverage plot with better visualization.
        
        Args:
            rssi_grids: List of 2D RSSI grids for each AP
            ap_locations: Dictionary of AP names and coordinates
            points: List of (x, y) coordinates for plot extent
            plots_dir: Directory to save plots
        """
        print("Generating enhanced combined coverage plot...")
        try:
            os.makedirs(plots_dir, exist_ok=True)
            
            # Calculate combined coverage (maximum signal at each point)
            combined_grid = np.max(np.stack(rssi_grids), axis=0)
            
            # Determine extent from points
            x_unique = np.unique([p[0] for p in points])
            y_unique = np.unique([p[1] for p in points])
            extent = (float(x_unique.min()), float(x_unique.max()), float(y_unique.min()), float(y_unique.max()))
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            im = plt.imshow(combined_grid, extent=extent, origin='lower',
                           cmap='RdYlBu_r', vmin=-90, vmax=-40)
            
            plt.title('Combined Coverage Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('X (meters)', fontsize=12)
            plt.ylabel('Y (meters)', fontsize=12)
            
            # Draw materials overlay
            self._draw_regions_overlay(plt.gca())
            
            # Draw all AP locations
            for ap_name, ap_coords in ap_locations.items():
                if len(ap_coords) >= 2:
                    plt.plot(ap_coords[0], ap_coords[1], 'X', color='red', markersize=15, markeredgewidth=2)
                    plt.text(ap_coords[0] + 0.2, ap_coords[1] + 0.2, ap_name, 
                            color='red', fontsize=10, fontweight='bold')
            
            plt.colorbar(im, label='Signal Strength (dBm)', shrink=0.8)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            
            output_filepath = os.path.join(plots_dir, 'coverage_combined_enhanced.png')
            plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Enhanced combined coverage plot saved to {output_filepath}")
            
        except Exception as e:
            print(f"Error in plot_combined_coverage_enhanced: {str(e)}")

    def plot_signal_quality_analysis(self, rssi_grids: List[np.ndarray], 
                                    ap_locations: Dict[str, Tuple[float, float]], 
                                    points: List[Tuple[float, float]], 
                                    plots_dir: str):
        """
        Generate signal quality analysis plots including coverage statistics.
        
        Args:
            rssi_grids: List of 2D RSSI grids for each AP
            ap_locations: Dictionary of AP names and coordinates
            points: List of (x, y) coordinates
            plots_dir: Directory to save plots
        """
        print("Generating signal quality analysis plots...")
        try:
            os.makedirs(plots_dir, exist_ok=True)
            
            # Calculate combined coverage
            combined_grid = np.max(np.stack(rssi_grids), axis=0)
            combined_flat = combined_grid.flatten()
            
            # Calculate coverage statistics
            excellent_threshold = -45  # dBm
            good_threshold = -67       # dBm
            
            excellent_coverage = np.sum(combined_flat >= excellent_threshold)
            good_coverage = np.sum((combined_flat >= good_threshold) & (combined_flat < excellent_threshold))
            poor_coverage = np.sum(combined_flat < good_threshold)
            total_points = len(combined_flat)
            
            # Create coverage pie chart
            plt.figure(figsize=(10, 6))
            coverage_data = [excellent_coverage, good_coverage, poor_coverage]
            coverage_labels = [
                f'Excellent\n(≥{excellent_threshold} dBm)\n{excellent_coverage/total_points*100:.1f}%',
                f'Good\n({good_threshold} to {excellent_threshold} dBm)\n{good_coverage/total_points*100:.1f}%',
                f'Poor\n(<{good_threshold} dBm)\n{poor_coverage/total_points*100:.1f}%'
            ]
            colors = ['green', 'yellow', 'red']
            
            plt.pie(coverage_data, labels=coverage_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Signal Quality Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'signal_quality_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create signal strength histogram
            plt.figure(figsize=(10, 6))
            plt.hist(combined_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=excellent_threshold, color='green', linestyle='--', linewidth=2, label=f'Excellent ({excellent_threshold} dBm)')
            plt.axvline(x=good_threshold, color='orange', linestyle='--', linewidth=2, label=f'Good ({good_threshold} dBm)')
            plt.title('Signal Strength Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Signal Strength (dBm)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'signal_strength_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create AP performance comparison
            plt.figure(figsize=(10, 6))
            ap_names = list(ap_locations.keys())
            mean_signals = [np.mean(grid) for grid in rssi_grids]
            
            bars = plt.bar(ap_names, mean_signals, color='lightcoral', alpha=0.8)
            plt.title('AP Performance Comparison', fontsize=14, fontweight='bold')
            plt.xlabel('Access Points')
            plt.ylabel('Mean Signal Strength (dBm)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, mean_signals):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'ap_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Signal quality analysis plots generated successfully.")
            
        except Exception as e:
            print(f"Error in plot_signal_quality_analysis: {str(e)}")
