import numpy as np

def select_target_random_block(occupied, H):
    """
    Randomly selects a target block from the set of non-full blocks.

    Parameters:
    - occupied (np.ndarray): Array of shape (Q,), number of agents in each block
    - H (int): Capacity of each block

    Returns:
    - index (int) of selected target block, or None if all blocks are full
    """
    candidates = np.where(occupied < H)[0]
    if len(candidates) == 0:
        return None  # no valid target
    return np.random.choice(candidates)


def select_target_random_cell(occupied, H):
    """
    Randomly selects a target block from the set of non-full blocks.

    Parameters:
    - occupied (np.ndarray): Array of shape (Q,), number of agents in each block
    - H (int): Capacity of each block

    Returns:
    - index (int) of selected target block, or None if all blocks are full
    """
    total_cells = H * occupied.shape[0] - np.sum(occupied)
    if total_cells == 0:
        return None  # no agents to select

    # Normalize to get sampling probabilities
    probs = (H - occupied) / total_cells

    # Weighted random choice over block indices
    return np.random.choice(len(occupied), p=probs)

