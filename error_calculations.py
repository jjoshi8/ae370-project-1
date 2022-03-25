import numpy as np

from numerical_methods import ivp_yoshida

def ivp_yoshida_error(u_0, T, delta_t, delta_t_baseline):

    """
    Implements the error in the predicted final system state for the 4th-order Yoshida integrator.
  
    Parameters
    ----------
    u_0 : array
        1 x N array defining the initial state vector u_0
    T : float_like
        Final time T
    delta_t : float_like
        Time step size where delta_t = t_{k+1} - t_k
        
    Returns
    -------
    err : float
        Error calculated as ||u_final_predicted - u_final_actual|| / ||u_final_actual||
        
    """

    # get full K x N arrays
    u_delta_t, times = ivp_yoshida(u_0, T, delta_t)
    u_delta_t_baseline, times_baseline = ivp_yoshida(u_0, T, delta_t_baseline)

    # extract only the final 1 x N array
    u_final_delta_t = u_delta_t[-1]
    u_final_delta_t_baseline = u_delta_t_baseline[-1]

    # compute the error
    err = la.norm(u_final_delta_t - u_final_delta_t_baseline) / la.norm(u_final_delta_t_baseline)

    return err

