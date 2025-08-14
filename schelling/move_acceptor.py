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


def build_marginal_table(H, utility_fn):
    rhos = np.arange(H + 1) / H
    u = np.vectorize(utility_fn)(rhos)
    g = H * rhos * u
    m = g[1:] - g[:-1]
    return m

def accept_metropolis(from_idx, to_idx, occupied, H, utility_fn, alpha=0.0, T=0.01, m_tab=None):
    delta_u = utility_fn((occupied[to_idx] + 1) / H) - utility_fn(occupied[from_idx] / H)
    G = delta_u

    if alpha:
        n_to   = occupied[to_idx]
        n_from = occupied[from_idx]


        if m_tab is not None:
            delta_U = m_tab[n_to] - m_tab[n_from - 1]
        else:
            def g(n):
                r = n / H
                return H * r * utility_fn(r)
            delta_U = (g(n_to + 1) - g(n_to)) + (g(n_from - 1) - g(n_from))

        G += alpha * (delta_U - delta_u)

    if T == 0:
        return G > 0

    P = 1.0 / (1.0 + np.exp(-np.clip(G / T, -60, 60)))
    return np.random.rand() < P
