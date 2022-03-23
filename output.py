import numpy as np
import matplotlib.pyplot as plt

from numerical_methods import ivp_yoshida
from numerical_methods import runge_kutta

# initialize radius of mars [km]
rm = 3396.2

# define initial states for bodies [km] and [km/s]
u_0_sat = np.array([20428, 0, 0, 0.33, 0.6, 0]) 
u_0_mars = np.array([0, 0, 0, 0, 0, 0])
u_0_pho = np.array([-1.05*rm, -0.676*rm, -2.47*rm, 1, 1, 1])
u_0_dei = np.array([5.89*rm, 3.31*rm, 1.46*rm, 1, 1, 1])
u_0_sun = np.array([10000*rm, 10000*rm, 10000*rm, 1, 1, 1])
u_0 = np.array([u_0_sat, u_0_mars, u_0_pho, u_0_dei, u_0_sun])

# define time step [s]
delta_t = 10

# define time [s]
T = 500000

# retrieve states using the Yoshida method
u_predicted, times = ivp_yoshida(u_0, T, delta_t)

plt.plot(u_predicted[:, 0, 0], u_predicted[:, 0, 1])
plt.show()

# retrieve final state for satellite
u_final_sat = u_predicted[-1][0, 0:3]

print(u_final_sat)