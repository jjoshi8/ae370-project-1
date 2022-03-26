import numpy as np

from f_true import f_true

def ivp_yoshida(u_0, T, delta_t, n):

    """
    Implements the predicted system evolution over time using the 4th-order Yoshida integration method.

    Parameters
    ----------
    u_0 : array
        1 x N array defining the initial state vector u_0
    T : float_like
        Final time T
    delta_t : float_like
        Time step size where delta_t = t_{k+1} - t_k
    n : integer
        # of bodies in the simulation
        
    Returns
    -------
    u : array
        K x N array of the predicted states where K = number of time steps
    times : array_like
        Length K vector containing the times t corresponding to time steps
        
    """

    # initialize u and time arrays
    K = int(T / delta_t) + 1
    n_size = int(u_0.shape[0])
    N_size = int(u_0.shape[1])
    u = np.zeros((K, n_size, N_size), dtype=float)
    times = np.linspace(0, T, K)

    # loop through Yoshida integrator computations
    u[0, :] = u_0
    for k in range(1, K):
        u[k] = yoshida(u[k-1], delta_t, n)

    return u, times

def yoshida(u_k, delta_t, n):

    """
    Implements the 4th-order Yoshida integrator to compute the predicted next state.
    
    Parameters
    ----------
    u_k : array
        Current state vector u_k 
    delta_t : float_like
        Time step size where delta_t = t_{k+1} - t_k
    n : integer
        # of bodies in the simulation
        
    Returns
    -------
    u_kplus1 : array
        1 x N array of the predicted next state vector u_{k+1}
        
    """

    # gather initial position and velocity vectors
    r_k = u_k[:, 0:3]
    v_k = u_k[:, 3:6]

    # define constants
    w0 = - (2 ** (1/3)) / (2 - (2 ** (1/3)))
    w1 = 1 / (2 - (2 ** (1/3)))
    c1 = w1 / 2
    c2 = (w0 + w1) / 2
    c3 = (w0 + w1) / 2
    c4 = w1 / 2
    d1 = w1
    d2 = w0
    d3 = w1
    
    # compute first position equation
    r_k1 = r_k + c1 * v_k * delta_t
    
    # compute acceleration at r_k1
    u_k1_temp = u_k
    u_k1_temp[:, 0:3] = r_k1
    a_k1 = f_true(u_k1_temp, n)[:, 3:6]

    # compute first velocity equation
    v_k1 = v_k + d1 * a_k1 * delta_t

    # compute second position equation
    r_k2 = r_k1 + c2 * v_k1 * delta_t

    # compute acceleration at r_k2
    u_k2_temp = u_k
    u_k2_temp[:, 0:3] = r_k2
    a_k2 = f_true(u_k2_temp, n)[:, 3:6]

    # compute second velocity equation
    v_k2 = v_k1 + d2 * a_k2 * delta_t

    # compute third position equation
    r_k3 = r_k2 + c3 * v_k2 * delta_t

    # compute acceleration at r_k3
    u_k3_temp = u_k
    u_k3_temp[:, 0:3] = r_k3
    a_k3 = f_true(u_k3_temp, n)[:, 3:6]

    # compute third velocity equation
    v_k3 = v_k2 + d3 * a_k3 * delta_t

    # compute next position and velocity vectors
    u_kplus1 = np.zeros(u_k.shape, dtype=float)
    u_kplus1[:, 0:3] = r_k3 + c4 * v_k3 * delta_t
    u_kplus1[:, 3:6] = v_k3

    return u_kplus1

def runge_kutta(u_k, delta_t):

    """
    Implements the Runge-Kutta method to compute the predicted next state.
    
    Parameters
    ----------
    u_k : array
        Current state vector u_k 
    delta_t : float_like
        Time step size where delta_t = t_{k+1} - t_k
        
    Returns
    -------
    u_kplus1 : array
        1 x N array of the predicted next state vector u_{k+1}
        
    """

    u_kplus1 = u_k

    return u_kplus1
