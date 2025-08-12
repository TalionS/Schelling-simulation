import numpy as np

def select_agent_random(occupied):
    """
    Randomly selects an agent uniformly from the whole population.

    Parameters:
    - occupied: np.array of shape (Q,), number of agents in each block

    Returns:
    - index (int) of selected agent's block
    """
    total_agents = np.sum(occupied)
    if total_agents == 0:
        return None  # no agents to select

    # Normalize to get sampling probabilities
    probs = occupied / total_agents

    # Weighted random choice over block indices
    return np.random.choice(len(occupied), p=probs)


def select_agent_random_with_types(agents):
    idx = np.random.randint(0, len(agents))
    return idx, agents[idx][0], agents[idx][1]