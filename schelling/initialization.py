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


import numpy as np

def init_random_with_types(Q, H, rho0, p):
    """
    Randomly initialize a city with egoists and altruists.

    Parameters:
    - Q (int): Number of neighborhoods (blocks)
    - H (int): Capacity of each neighborhood
    - rho0 (float): Initial average density (0 < rho0 <= 1)
    - p (float): Proportion of altruists in the population (0 <= p <= 1)

    Returns:
    - occupied (np.ndarray): length-Q array, number of agents in each block
    - agents (list of tuple): each element is (block_id, type),
                              where type=0 for egoist, 1 for altruist
    """
    # Total number of agents
    N = int(Q * H * rho0)

    # Number of altruists
    N_altruist = int(round(p * N))
    # Number of egoists
    N_egoist = N - N_altruist

    occupied = np.zeros(Q, dtype=int)
    agents = []

    for _ in range(N_altruist):
        q = np.random.randint(Q)
        while occupied[q] >= H:
            q = np.random.randint(Q)
        occupied[q] += 1
        agents.append([q, 1]) 

    for _ in range(N_egoist):
        q = np.random.randint(Q)
        while occupied[q] >= H:
            q = np.random.randint(Q)
        occupied[q] += 1
        agents.append([q, 0])  

    return occupied.copy(), agents