import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patheffects import withStroke
import os


def plot_ultra_advanced_coverage_and_aps(
    floor_plan_img_path,
    ap_locations,
    signal_grid,
    x_coords,
    y_coords,
    output_path_prefix,
    wall_lines=None,
    room_polygons=None,
    dpi=400,
    show=True
):
    """
    Ultra-advanced WiFi AP placement and coverage visualization.
    - floor_plan_img_path: path to floor plan image (JPG/PNG)
    - ap_locations: dict {APn: (x, y, ...)}
    - signal_grid: 2D np.ndarray (signal strength)
    - x_coords, y_coords: 1D arrays for grid axes
    - output_path_prefix: base path for saving (no extension)
    - wall_lines: list of ((x1, y1), (x2, y2))
    - room_polygons: list of [(x1, y1), (x2, y2), ...]
    - dpi: output resolution
    - show: whether to display plot interactively
    """
    # Load floor plan image
    img = plt.imread(floor_plan_img_path)
    img_extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

    # Plot floor plan
    ax.imshow(img, extent=img_extent, aspect='auto', alpha=0.6, zorder=0)

    # Plot coverage heatmap
    cmap = plt.get_cmap('coolwarm')
    vmin, vmax = -90, -30
    im = ax.imshow(
        signal_grid,
        extent=img_extent,
        origin='lower',
        cmap=cmap,
        alpha=0.55,
        vmin=vmin,
        vmax=vmax,
        zorder=1
    )

    # Plot walls
    if wall_lines:
        for (x1, y1), (x2, y2) in wall_lines:
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=3, alpha=0.7, zorder=3)

    # Plot rooms
    if room_polygons:
        for poly in room_polygons:
            patch = mpatches.Polygon(poly, closed=True, fill=False, edgecolor='gray', linewidth=2, alpha=0.5, zorder=2)
            ax.add_patch(patch)

    # AP marker styles
    ap_colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
        '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
    ]
    marker_styles = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '8']
    
    # Plot APs
    for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
        x, y = ap_coords[:2]
        color = ap_colors[i % len(ap_colors)]
        marker = marker_styles[i % len(marker_styles)]
        ax.scatter(x, y, s=600, c=color, marker=marker, edgecolors='black', linewidths=2, zorder=10)
        ax.text(
            x, y, f'{i+1}',
            fontsize=22, fontweight='bold', color='white',
            ha='center', va='center', zorder=11,
            path_effects=[withStroke(linewidth=4, foreground='black')]
        )
        ax.text(
            x, y-1.5, ap_name,
            fontsize=13, fontweight='bold', color='black',
            ha='center', va='top', zorder=12,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=1, alpha=0.8)
        )

    # Title and labels
    ax.set_title('Ultra-Advanced WiFi Coverage and AP Placement', fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel('X (meters)', fontsize=16)
    ax.set_ylabel('Y (meters)', fontsize=16)
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(y_coords[0], y_coords[-1])
    ax.set_aspect('equal')
    ax.grid(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label('Signal Strength (dBm)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # AP legend
    legend_handles = [
        mpatches.Patch(color=ap_colors[i % len(ap_colors)], label=f'{ap_name}')
        for i, ap_name in enumerate(ap_locations.keys())
    ]
    ax.legend(handles=legend_handles, title='Access Points', fontsize=13, title_fontsize=15, loc='upper right', bbox_to_anchor=(1.18, 1))

    # Save in multiple formats
    for ext in ['png', 'svg', 'pdf']:
        out_path = f'{output_path_prefix}_ultra.{ext}'
        fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close(fig) 