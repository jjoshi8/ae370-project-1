import numpy as np

def f_true_baseline(u, m_satellite=3300):

    """
    Returns the output dynamics vector for a given u and t.

    Parameters
    ----------
    u : array-like
        Current state vector [r_x, r_y, r_z, v_x, v_y, v_z]
    m_satellite : float_like (kg)
        Mass of satellite/vehicle orbiting Mars

    Constants
    ---------
    m_mars: float_like (kg)
        Mass of Mars

    Returns
    -------
    u_dot : array
        Vector of output dynamics for a given u and t.
    
    """


    m_mars = 6.39e23

    return 0

def f_true_moons(u, m_satellite=3300, m_mars=6.39e23, m_phobos=10.8e15, m_deimos=1.8):

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

    Returns
    -------
    u_dot : array
        Vector of output dynamics for a given u and t.
    
    """

    return

def f_true_sun(u, m_satellite=3300, m_mars=6.39e23, m_sun=1.9891e30):

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
    m_sun : float-like (kg)
        Mass of the Sun

    Returns
    -------
    u_dot : array
        Vector of output dynamics for a given u and t.
    
    """

    return

def f_true_all(u, m_satellite=3300, m_mars=6.39e23, m_phobos=10.8e15, m_deimos=1.8, m_sun=1.9891e30):

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

    return 
