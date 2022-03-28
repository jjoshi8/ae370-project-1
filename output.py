import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

from numerical_methods import ivp_symplectic
from numerical_methods import ivp_runge_kutta
from error_calculations import ivp_symplectic_error
from error_calculations import ivp_runge_kutta_error

plot_nominal = False
plot_error = False
plot_compare_n = False
plot_compare_n_error = False
plot_compare_num_method = True

# define Mars' radius [km]
rm = 3396.2

# define initial states for bodies [km] and [km/s]
u_0_sat = np.array([20428, 0, 0, 0, 1.448, 0])
u_0_mars = np.array([0, 0, 0, 0, 0, 0])
u_0_pho = np.array([-3.3407e3, -8.6799e3, 9.6013e2, 1.7860, -0.75438, -0.91419])
u_0_dei = np.array([2.1332e4, 2.5385e3, -9.4214e3, -0.092389, 1.3394, 0.15256])
u_0_sun = np.array([1.1299e8, 1.9818e8, 1.3819e6, -2.1959e1, 9.9224, 0.74659])
u_0 = np.array([u_0_sat, u_0_mars, u_0_pho, u_0_dei, u_0_sun])

# define time step and time[s]
delta_t = 15
T = 88642

# plot nominal trajectory
if plot_nominal:

    # define n for n-body system
    n = 5

    # retrieve states using the specified numerical method
    u_predicted, times = ivp_runge_kutta(u_0, T, delta_t, n)

    # plot Mars centered at (0, 0, 0)
    ax = plt.axes(projection='3d')
    u, v = np.mgrid[0:2*np.pi:2000j, 0:np.pi:2000j]
    x = rm * np.cos(u) * np.sin(v)
    y = rm * np.sin(u) * np.sin(v)
    z = rm * np.cos(v)
    ax.plot_wireframe(x, y, z, color='brown')

    # plot satellite position relative to Mars
    x_sat_rel = u_predicted[:, 0, 0] - u_predicted[:, 1, 0]
    y_sat_rel = u_predicted[:, 0, 1] - u_predicted[:, 1, 1]
    z_sat_rel = u_predicted[:, 0, 2] - u_predicted[:, 1, 2]
    ax.plot(x_sat_rel, y_sat_rel, z_sat_rel)

    # set axis limits for symmetry
    ax.set_xlim([-25000, 25000])
    ax.set_ylim([-25000, 25000])
    ax.set_zlim([-25000, 25000])

    # show plot
    plt.show()

# plot error results
if plot_error:

    # define n for n-body system
    n = 2

    # compute error for different time steps
    delta_ts = [1e2, 2.5e2, 5e2, 1e3]
    delta_t_baseline = 1

    # loop through delta_t values
    delta_t_errors = []
    for delta_t in delta_ts:
        delta_t_errors.append(ivp_runge_kutta_error(u_0, T, delta_t, delta_t_baseline, n))

    # plot error results
    ax = plt.axes()
    ax.loglog(delta_ts, delta_t_errors, 'b', marker='o')
    ax.set_xlabel("$\Delta$t")
    ax.set_ylabel("Error")
    ax.set_title("4th-Order Runge-Kutta Method")

    # show plot
    plt.show()

# plot comparison between n-body problems
if plot_compare_n:

    # retrieve states using the specified numerical method
    u_predicted_2, times = ivp_runge_kutta(u_0, T, delta_t, 2, m_phobos=6.39e23)
    u_predicted_5, times = ivp_runge_kutta(u_0, T, delta_t, 5, m_phobos=6.39e23)

    # plot Mars centered at (0, 0, 0)
    ax = plt.axes(projection='3d')
    u, v = np.mgrid[0:2*np.pi:2000j, 0:np.pi:2000j]
    x = rm * np.cos(u) * np.sin(v)
    y = rm * np.sin(u) * np.sin(v)
    z = rm * np.cos(v)
    ax.plot_wireframe(x, y, z, color='brown')

    # plot satellite position relative to Mars
    x_sat_rel_2 = u_predicted_2[:, 0, 0] - u_predicted_2[:, 1, 0]
    y_sat_rel_2 = u_predicted_2[:, 0, 1] - u_predicted_2[:, 1, 1]
    z_sat_rel_2 = u_predicted_2[:, 0, 2] - u_predicted_2[:, 1, 2]
    ax.plot(x_sat_rel_2, y_sat_rel_2, z_sat_rel_2, 'b-', label='2-body')

    # plot satellite position relative to Mars
    x_sat_rel_5 = u_predicted_5[:, 0, 0] - u_predicted_5[:, 1, 0]
    y_sat_rel_5 = u_predicted_5[:, 0, 1] - u_predicted_5[:, 1, 1]
    z_sat_rel_5 = u_predicted_5[:, 0, 2] - u_predicted_5[:, 1, 2]
    ax.plot(x_sat_rel_5, y_sat_rel_5, z_sat_rel_5, 'm--', label='5-body')

    # set axis limits for symmetry
    ax.set_xlim([-2.5e4, 2.5e4])
    ax.set_ylim([-2.5e4, 2.5e4])
    ax.set_zlim([-2.5e4, 2.5e4])

    # set labels
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('X [km]', labelpad=10, rotation=0)
    ax.set_ylabel('Y [km]', labelpad=15, rotation=0)
    ax.set_zlabel('Z [km]', labelpad=10, rotation=90)

    # show plot
    ax.legend()
    plt.show()

# plot comparison between n-body problems
if plot_compare_n_error:

    # retrieve states using the specified numerical method
    u_predicted_2, times = ivp_runge_kutta(u_0, T, delta_t, 2, m_phobos=7.3483e22)
    u_predicted_5, times = ivp_runge_kutta(u_0, T, delta_t, 5, m_phobos=7.3483e22)

    # plot satellite position relative to Mars
    ax = plt.axes()
    x_sat_rel_2 = u_predicted_2[:, 0, 0] - u_predicted_2[:, 1, 0]
    x_sat_rel_5 = u_predicted_5[:, 0, 0] - u_predicted_5[:, 1, 0]
    y_sat_rel_2 = u_predicted_2[:, 0, 1] - u_predicted_2[:, 1, 1]
    y_sat_rel_5 = u_predicted_5[:, 0, 1] - u_predicted_5[:, 1, 1]
    z_sat_rel_2 = u_predicted_2[:, 0, 2] - u_predicted_2[:, 1, 2]
    z_sat_rel_5 = u_predicted_5[:, 0, 2] - u_predicted_5[:, 1, 2]
    x_error = abs(x_sat_rel_2 - x_sat_rel_5)
    y_error = abs(y_sat_rel_2 - y_sat_rel_5)
    z_error = abs(z_sat_rel_2 - z_sat_rel_5)
    total_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
    ax.plot(times, x_error, 'm', label='X-Error')
    ax.plot(times, y_error, 'g', label='Y-Error')
    ax.plot(times, z_error, 'b', label='Z-Error')
    ax.plot(times, total_error, 'k', label='Total Error')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [km]')

    # show plot
    ax.legend()
    plt.show()

# plot error results
if plot_compare_num_method:

    # define n for n-body system
    n = 5

    # compute error for different time steps
    delta_ts = [1, 5, 10, 15, 30, 60]

    # loop through delta_t values
    run_time_runge_kutta = []
    run_time_sympectic = []
    for delta_t in delta_ts:
        time1 = time.time()
        ivp_runge_kutta(u_0, T, delta_t, n)
        time2 = time.time()
        ivp_symplectic(u_0, T, delta_t, n)
        time3 = time.time()
        run_time_runge_kutta.append(time2 - time1)
        run_time_sympectic.append(time3 - time2)

    # plot error results
    ax = plt.axes()
    ax.plot(delta_ts, run_time_runge_kutta, 'b', label='Runge-Kutta', marker='o')
    ax.plot(delta_ts, run_time_sympectic, 'm', label='3rd-Order Symplectic', marker='o')
    ax.set_xlabel("$\Delta$t")
    ax.set_ylabel("Runtime [s]")

    # show plot
    ax.legend()
    plt.show()