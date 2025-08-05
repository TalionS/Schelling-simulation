def stop_after_fixed_steps(current_step, max_steps):
    """
    Stop criterion: stop the simulation after a fixed number of steps.

    Parameters:
    - current_step (int): Current simulation step (starting from 0)
    - max_steps (int): Maximum number of allowed steps

    Returns:
    - True if the simulation should stop, False otherwise
    """
    return current_step >= max_steps