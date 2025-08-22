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


def altruist_utility(rho):
    return np.where(rho <= 0.5, 4 * rho, 2 * (1 - 2 * rho))


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


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def plot_block_agents(rows, cols, H, agents, title='',
                      seed=None,
                      egoist_color='#1f77b4', altruist_color='#d62728', empty_color='#DDDDDD',
                      save_path=None, dpi=200, show=True):
    """
    Plot a city where each block is split into H cells; agents occupy random empty cells.
    Different colors denote agent types: egoist(0) vs altruist(1).

    Parameters
    ----------
    rows, cols : int
        City grid size (rows x cols blocks).
    H : int
        Capacity (cells) per block.
    agents : list[tuple]
        Each element is (block_id, type), where type=0 (egoist) or 1 (altruist).
        block_id in [0, rows*cols-1].
    title : str
        Figure title.
    seed : int or None
        Random seed for reproducible placement inside blocks.
    egoist_color, altruist_color, empty_color : str
        Colors for plotting.
    save_path : str or None
        Full path to save figure (e.g., "outputs/city_run1.png"). If None, do not save.
        Directory will be created if it doesn't exist.
    dpi : int
        DPI used when saving.
    show : bool
        Whether to display the figure window.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib Figure and Axes objects.
    """
    Q = rows * cols
    if any((b < 0 or b >= Q or t not in (0, 1)) for b, t in agents):
        raise ValueError("agents contains invalid (block_id, type).")

    rng = np.random.default_rng(seed)

    # tile layout inside each block to fit H cells
    br = int(np.floor(np.sqrt(H)))      # rows per block tile
    bc = int(np.ceil(H / br))           # cols per block tile
    assert br * bc >= H

    big_h = rows * br
    big_w = cols * bc

    canvas = np.full((big_h, big_w), -1, dtype=int)

    # group agent types by block
    per_block = [[] for _ in range(Q)]
    for b, t in agents:
        per_block[b].append(t)

    # place agents randomly within each block tile
    for q in range(Q):
        types = per_block[q]
        if len(types) > H:
            types = types[:H]

        block_row = q // cols
        block_col = q % cols
        r0 = block_row * br
        c0 = block_col * bc

        sub_positions = [(i, j) for i in range(br) for j in range(bc)]
        rng.shuffle(sub_positions)

        types = list(types)
        rng.shuffle(types)

        for k, t in enumerate(types):
            rr, cc = sub_positions[k]
            canvas[r0 + rr, c0 + cc] = t

    # colors and normalization
    cmap = ListedColormap([empty_color, egoist_color, altruist_color])
    boundaries = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(boundaries, cmap.N)

    fig_w = max(6, cols * 2)
    fig_h = max(6, rows * 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(canvas, cmap=cmap, norm=norm, interpolation='nearest')

    # draw thick block grid lines
    for r in range(rows + 1):
        ax.axhline(r * br - 0.5, color='white', linewidth=2)
    for c in range(cols + 1):
        ax.axvline(c * bc - 0.5, color='white', linewidth=2)

    # thin cell grid
    ax.set_xticks(np.arange(-0.5, big_w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, big_h, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.6)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    legend_patches = [
        Patch(facecolor=egoist_color, edgecolor='none', label='Egoist (0)'),
        Patch(facecolor=altruist_color, edgecolor='none', label='Altruist (1)'),
        Patch(facecolor=empty_color, edgecolor='none', label='Empty')
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True)

    plt.tight_layout()

    # save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


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
