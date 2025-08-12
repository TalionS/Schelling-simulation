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


def select_target_max_utility(occupied, H, utility_fn):
    """
    Select the target block that yields the maximum utility after one agent moves in.

    Parameters:
    - occupied (np.ndarray): shape (Q,), current number of agents in each block
    - H (int): block capacity
    - utility_fn (function): utility function u(ρ), where ρ is density ∈ [0, 1]

    Returns:
    - index (int) of the selected target block, or None if all are full
    """
    candidates = np.where(occupied < H)[0]
    if len(candidates) == 0:
        return None

    # Compute potential utility after moving in
    candidate_densities = (occupied[candidates] + 1) / H
    candidate_utils = utility_fn(candidate_densities)

    # Select block with maximum utility
    max_util = np.max(candidate_utils)
    max_indices = np.where(candidate_utils == max_util)[0]

    chosen_idx = np.random.choice(max_indices)
    return candidates[chosen_idx]

