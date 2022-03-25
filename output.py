import numpy as np
import matplotlib.pyplot as plt

from numerical_methods import ivp_yoshida
from numerical_methods import runge_kutta

# initialize radius of mars [km]
rm = 3396.2

# define initial states for bodies [km] and [km/s]
u_0_sat = np.array([5000, 0, 0, 0, 3, 0])
u_0_mars = np.array([0, 0, 0, 0, 0, 0])
u_0_pho = np.array([-1.05*rm, -0.676*rm, -2.47*rm, 1, 1, 1])
u_0_dei = np.array([5.89*rm, 3.31*rm, 1.46*rm, 1, 1, 1])
u_0_sun = np.array([10000*rm, 10000*rm, 10000*rm, 1, 1, 1])
u_0 = np.array([u_0_sat, u_0_mars, u_0_pho, u_0_dei, u_0_sun])

# define time step [s]
delta_t = 15

# define time [s]
T = 111 * 60 * 2

# retrieve states using the Yoshida method
u_predicted, times = ivp_yoshida(u_0, T, delta_t)
print(u_predicted[320, 0, 3:5]) # -1.41067131e+06

ax = plt.axes(projection='3d')
u, v = np.mgrid[0:2*np.pi:2000j, 0:np.pi:2000j]
x = rm * np.cos(u) * np.sin(v)
y = rm * np.sin(u) * np.sin(v)
z = rm * np.cos(v)
ax.plot_wireframe(x, y, z, color='brown')
ax.plot(u_predicted[:, 0, 0], u_predicted[:, 0, 1], u_predicted[:, 0, 2])

# plt.plot(u_predicted[:, 0, 0], u_predicted[:, 0, 1])
# plt.plot(times, u_predicted[:, 0, 1])
plt.show()
# plt.show(block=False)
# plt.pause(5)
# plt.close()

# retrieve final state for satellite
u_final_sat = u_predicted[-1][0, 0:3]

print(u_final_sat)