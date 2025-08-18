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


def select_target_max_utility(occupied, H, utility_fn, the_same_utility_function=True):
    """
    Select the target block that yields the maximum (post-move-in) utility.

    Parameters
    ----------
    occupied : np.ndarray, shape (Q,)
        Current number of agents in each block.
    H : int
        Capacity per block.
    utility_fn :
        - If the_same_utility_function == True:
            a single callable u(rho) valid for all blocks.
        - If the_same_utility_function == False:
            a sequence (list/tuple) of length Q with per-block callables u_q(rho).
    the_same_utility_function : bool
        Whether all blocks share the same utility function.

    Returns
    -------
    int or None
        Index of the selected target block, or None if all blocks are full.
    """
    candidates = np.where(occupied < H)[0]
    if candidates.size == 0:
        return None

    if the_same_utility_function:
        candidate_densities = (occupied[candidates] + 1) / H
        try:
            candidate_utils = utility_fn(candidate_densities)
        except Exception:
            candidate_utils = np.array([utility_fn(r) for r in candidate_densities])
    else:
        candidate_utils = np.array([
            utility_fn[q]((occupied[q] + 1) / H) for q in candidates
        ])

    max_util = np.max(candidate_utils)
    max_indices = np.where(candidate_utils == max_util)[0]
    chosen_idx = np.random.choice(max_indices)
    return candidates[chosen_idx]
