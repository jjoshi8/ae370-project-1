import numpy as np
import matplotlib.pyplot as plt

from numerical_methods import ivp_yoshida
from numerical_methods import runge_kutta
from error_calculations import ivp_yoshida_error

# initialize radius of mars [km]
rm = 3396.2

# define initial states for bodies [km] and [km/s]
u_0_sat = np.array([20428, 0, 0, 0, 1.448, 0])
u_0_mars = np.array([0, 0, 0, 0, 0, 0])
u_0_pho = np.array([-3.3407e3, -8.6799e3, 9.6013e2, 1.7860, -0.75438, -0.91419])
u_0_dei = np.array([2.1332e4, 2.5385e3, -9.4214e3, -0.092389, 1.3394, 0.15256])
u_0_sun = np.array([1.1299e8, 1.9818e8, 1.3819e6, -2.1959e1, 9.9224, 0.74659])
u_0 = np.array([u_0_sat, u_0_mars, u_0_pho, u_0_dei, u_0_sun])

# define time step [s]
delta_t = 15

# define time [s]
T = 200000 #111 * 60 * 2

# retrieve states using the Yoshida method
n = 5
u_predicted, times = ivp_yoshida(u_0, T, delta_t, n)

ax = plt.axes(projection='3d')
u, v = np.mgrid[0:2*np.pi:2000j, 0:np.pi:2000j]
x = rm * np.cos(u) * np.sin(v)
y = rm * np.sin(u) * np.sin(v)
z = rm * np.cos(v)
ax.plot_wireframe(x, y, z, color='brown')
x_sat_rel = u_predicted[:, 0, 0] - u_predicted[:, 1, 0]
y_sat_rel = u_predicted[:, 0, 1] - u_predicted[:, 1, 1]
z_sat_rel = u_predicted[:, 0, 2] - u_predicted[:, 1, 2]
ax.plot(x_sat_rel, y_sat_rel, z_sat_rel)
ax.set_xlim([-25000, 25000])
ax.set_ylim([-25000, 25000])
ax.set_zlim([-25000, 25000])
plt.show()
# plt.show(block=False)
# plt.pause(5)
# plt.close()

# compute error for different time steps
# delta_ts = [10, 20, 40, 50, 100]
# delta_t_baseline = 1
delta_ts = [15, 25, 35, 50, 60]
delta_t_baseline = 1
# delta_ts = [5 * 10**-2, 2.5 * 10**-2, 1 * 10**-2, 5 * 10**-3, 2.5 * 10**-3, 1 * 10**-3, 5 * 10**-4]
# delta_t_baseline = 2.5 * 10**-4
yoshida_delta_t_errors = []

for delta_t in delta_ts:
    yoshida_delta_t_errors.append(ivp_yoshida_error(u_0, T, delta_t, delta_t_baseline, n))

ax = plt.axes()
ax.loglog(delta_ts, yoshida_delta_t_errors, 'b', marker='o')
ax.set_xlabel("$\Delta$t")
ax.set_ylabel("Error")
ax.set_title("4th-Order Yoshida Integrator")
ax.set_ylim([10**-15, 10**-8])


plt.show(block=False)
plt.pause(100)
plt.close()

# retrieve final state for satellite
u_final_sat = u_predicted[-1][0, 0:3]

print(u_final_sat)