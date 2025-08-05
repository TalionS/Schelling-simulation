import numpy as np
import matplotlib.pyplot as plt


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