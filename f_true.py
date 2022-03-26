import numpy as np
import numpy.linalg as la

def f_true(u, n):

    """
    Wrapper function that returns the true function of the desired n-body simulation.

    Parameters
    ----------
    u : array-like
        Current state vector [r_x, r_y, r_z, v_x, v_y, v_z]

    Returns
    -------
    u_dot : array
        Vector of output dynamics for a given u and t.

    """

    if n == 5:
        return f_true_all(u)

    if n == 2:
        return f_true_baseline(u)

    return f_true_baseline(u)

def f_true_baseline(u, m_satellite=3300, m_mars=6.39e23):

    """
    Returns the output dynamics vector for a given u and t.

    Parameters
    ----------
    u : array-like
        Current state vector [r_x, r_y, r_z, v_x, v_y, v_z]
    m_satellite : float_like (kg)
        Mass of satellite/vehicle orbiting Mars
    m_mars: float_like (kg)
        Mass of Mars

    Returns
    -------
    u_dot : array
        Vector of output dynamics for a given u and t.
    
    """

    # define constants (convert to km)
    G = 6.67 * (10 ** -11) * (10 ** -9)
    m = [m_satellite, m_mars]
    n = len(m)

    # define the maximum number of n-bodies for consistency
    max_n = 5

    # initialize u_dot array
    u_dot = np.zeros((max_n, 6), dtype=float)

    # compute velocity components
    u_dot[:n, 0] = u[:n, 3]
    u_dot[:n, 1] = u[:n, 4]
    u_dot[:n, 2] = u[:n, 5]

    # compute acceleration components
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            u_dot[i, 3] += G * m[j] * (u[j, 0] - u[i, 0]) / (la.norm(u[j, 0:3] - u[i, 0:3]) ** 3)
            u_dot[i, 4] += G * m[j] * (u[j, 1] - u[i, 1]) / (la.norm(u[j, 0:3] - u[i, 0:3]) ** 3)
            u_dot[i, 5] += G * m[j] * (u[j, 2] - u[i, 2]) / (la.norm(u[j, 0:3] - u[i, 0:3]) ** 3)

    return u_dot

def f_true_all(u, m_satellite=3300, m_mars=6.39e23, m_phobos=10.8e15, m_deimos=1.8e15, m_sun=1.9891e30):

    """
    Returns the output dynamics vector for a given u and t.

    Parameters
    ----------
    u : array-like
        Current state vector [r_x, r_y, r_z, v_x, v_y, v_z]
    m_satellite : float_like (kg)
        Mass of satellite/vehicle orbiting Mars
    m_mars: float_like (kg)
        Mass of Mars
    m_phobos : float-like (kg)
        Mass of Phobos
    m_deimos : float-like (kg)
        Mass of Deimos
    m_sun : float-like (kg)
        Mass of the Sun

    Returns
    -------
    u_dot : array
        Vector of output dynamics for a given u and t.
    
    """

    # define constants (convert to km)
    G = 6.67 * (10 ** -11) * (10 ** -9)
    m = [m_satellite, m_mars, m_phobos, m_deimos, m_sun]
    n = len(m)

    # initialize u_dot array
    u_dot = np.zeros((n, 6), dtype=float)

    # compute velocity components
    u_dot[:, 0] = u[:, 3]
    u_dot[:, 1] = u[:, 4]
    u_dot[:, 2] = u[:, 5]

    # compute acceleration components
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            u_dot[i, 3] += G * m[j] * (u[j, 0] - u[i, 0]) / (la.norm(u[j, 0:3] - u[i, 0:3]) ** 3)
            u_dot[i, 4] += G * m[j] * (u[j, 1] - u[i, 1]) / (la.norm(u[j, 0:3] - u[i, 0:3]) ** 3)
            u_dot[i, 5] += G * m[j] * (u[j, 2] - u[i, 2]) / (la.norm(u[j, 0:3] - u[i, 0:3]) ** 3)

    return u_dot
