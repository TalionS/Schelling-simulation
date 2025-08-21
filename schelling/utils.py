import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.tri import Triangulation


def triangle_utility(rho):
    """
    Triangle-shaped utility function that supports both scalar and array input.

    u(rho) = 2 * rho       if rho <= 0.5
             2 * (1 - rho) if rho > 0.5

    Parameters:
    - rho (float or np.ndarray): density value(s)

    Returns:
    - utility (float or np.ndarray): computed utility value(s)
    """
    rho = np.asarray(rho)  # convert to array if needed
    return np.where(rho <= 0.5, 2 * rho, 2 * (1 - rho))


def asymmetric_triangle_utility(x, peak_x=0.5, peak_y=1.0, left_y=0.0, right_y=0.0):
    """
    Asymmetric piecewise-linear utility function (triangle shape):
    
    - Rises linearly from (0, left_y) to (peak_x, peak_y)
    - Falls linearly from (peak_x, peak_y) to (1, right_y)

    Parameters:
    - x (float or np.ndarray): The density value(s), expected in [0, 1]
    - peak_x (float): x-location of the utility peak (0 < peak_x < 1)
    - peak_y (float): Maximum utility at the peak
    - left_y (float): Utility at density = 0
    - right_y (float): Utility at density = 1

    Returns:
    - u (float or np.ndarray): Utility value(s) corresponding to x
    """
    x = np.clip(x, 0, 1)  # Ensure x stays within [0, 1]
    x = np.asarray(x)

    # Compute slopes for the left and right segments
    slope_left = (peak_y - left_y) / peak_x
    slope_right = (right_y - peak_y) / (1 - peak_x)

    # Apply piecewise formula
    u = np.where(
        x <= peak_x,
        left_y + slope_left * x,
        peak_y + slope_right * (x - peak_x)
    )

    return u


def plot_density_heatmap(rows, cols, density, title='', xlabel='', ylabel=''):
    """
    Plots a heatmap of block densities.

    Parameters:
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - density (np.array): 1D array of densities per block, length <= rows * cols
    - title (str): Title of the heatmap.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    total_blocks = rows * cols
    if len(density) > total_blocks:
        raise ValueError(f"Grid size ({rows} × {cols}) is too small for density vector of length {len(density)}.")

    # Fill the rest with NaNs for better visualization if not a perfect grid
    padded = np.full(total_blocks, np.nan)
    padded[:len(density)] = density

    # Reshape to 2D grid
    grid = padded.reshape((rows, cols))

    plt.figure(figsize=(cols, rows))
    im = plt.imshow(grid, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)

    # Add value annotations
    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')

    plt.colorbar(im, label='Density')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=np.arange(cols))
    plt.yticks(ticks=np.arange(rows))
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_surface_from_points(x, y, z, *,
                             x_label='m', y_label='alpha', z_label='U*',
                             cmap='viridis', elev=30, azim=-60,
                             linewidth=0.2, edgecolor='k',
                             show_scatter=False, scatter_size=5):

    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    assert x.size == y.size == z.size, "x,y,z should have the same dimension"

    tri = Triangulation(x, y)  

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(tri, z, cmap=cmap, linewidth=linewidth,
                           antialiased=True, edgecolor=edgecolor)

    if show_scatter:
        ax.scatter(x, y, z, s=scatter_size, c=z, cmap=cmap, alpha=0.6)

    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_zlabel(z_label)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=z_label)
    plt.tight_layout()
    plt.show()


def plot_line_chart(x, y, title='Line Chart', xlabel='x-axis', ylabel='y-axis',
                    line_style='-', marker='o', color='b', label=None,
                    figsize=(8, 5), grid=True, legend=True):
    """
    Plot a 2D line chart.

    Parameters:
    - x, y: Lists or NumPy arrays for x and y coordinates
    - title (str): Title of the chart
    - xlabel (str): Label for the x-axis
    - ylabel (str): Label for the y-axis
    - line_style (str): Line style ('-', '--', ':', etc.)
    - marker (str): Marker style for points ('o', 's', '^', etc.)
    - color (str): Line color (e.g., 'b' for blue, 'r' for red)
    - label (str): Label for the legend (optional)
    - figsize (tuple): Size of the figure (width, height)
    - grid (bool): Whether to display gridlines
    - legend (bool): Whether to display a legend
    """
    # Create a new figure with the specified size
    plt.figure(figsize=figsize)

    # Plot the line with the given style, marker, color, and optional label
    plt.plot(x, y, linestyle=line_style, marker=marker, color=color, label=label)

    # Set chart title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Enable grid if specified
    if grid:
        plt.grid(True)

    # Show legend if specified and label is provided
    if legend and label is not None:
        plt.legend()

    # Display the plot
    plt.show()