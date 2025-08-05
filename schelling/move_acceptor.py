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