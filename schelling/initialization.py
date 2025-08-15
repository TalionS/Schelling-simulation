import numpy as np


def init_random(Q, H, rho0):
    """
    Randomly initialize a city by placing agents according to a given global density.

    Parameters:
    - Q (int): Number of neighborhoods (blocks)
    - H (int): Capacity of each neighborhood
    - rho0 (float): Initial average density (0 < rho0 <= 1)

    Returns:
    - occupied (np.ndarray): Array of length Q, indicating the number of agents in each neighborhood
    """
    # Total number of agents to place
    N = int(Q * H * rho0)

    # Initialize all blocks as empty
    occupied = np.zeros(Q, dtype=int)

    # Place each agent one by one into a randomly selected block
    for _ in range(N):
        q = np.random.randint(Q)
        # If the selected block is full, keep trying until a non-full block is found
        while occupied[q] >= H:
            q = np.random.randint(Q)
        occupied[q] += 1

    return occupied.copy()
