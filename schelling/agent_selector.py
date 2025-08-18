import numpy as np


def select_agent_random_with_types(agents):
    idx = np.random.randint(0, len(agents))
    return idx, agents[idx][0], agents[idx][1]