import numpy as np

def accept_if_personal_utility_improves(from_density, to_density, utility_fn):
    """
    Accept the move if the agent's personal utility increases.

    Parameters:
    - from_density (float): Current density of the block the agent is in
    - to_density (float): Density of the block the agent is considering moving to
    - utility_fn (callable): Function u(ρ) that returns utility given density ρ

    Returns:
    - True if move is accepted (Δu > 0), False otherwise
    """
    u_current = utility_fn(from_density)
    u_new = utility_fn(to_density)
    return u_new > u_current


def accept_metropolis(from_idx, to_idx, occupied, H, utility_fn, alpha=0.0, T=0.01):
    delta_u = utility_fn((occupied[to_idx] + 1) / H) - utility_fn(occupied[from_idx] / H)
    G = delta_u
    if alpha:
        tmp_occupied = occupied.copy()
        tmp_occupied[from_idx] -= 1
        tmp_occupied[to_idx] += 1

        rho_before = occupied / H
        rho_after = tmp_occupied / H

        u_before = np.array([utility_fn(r) for r in rho_before])
        u_after  = np.array([utility_fn(r) for r in rho_after])

        delta_U = H * np.sum(rho_after * u_after - rho_before * u_before)
    
        G += alpha * (delta_U - delta_u)

    if T == 0:
        return G > 0

    P = 1 / (1 + np.exp(-np.clip(G / T, -60, 60)))

    return np.random.rand() < P