import numpy as np
import matplotlib.pyplot as plt

v10 = np.load('y0_plot_v10.npz')
y0_v10 = v10.get('y0')
t_v10 = v10.get('time')

v15 = np.load('y0_plot_v15.npz')
y0_v15 = v15.get('y0')
t_v15 = v15.get('time')

v20 = np.load('y0_plot_v20.npz')
y0_v20 = v20.get('y0')
t_v20 = v20.get('time')

v20 = np.load('y0_plot_v20dark.npz')
y0_v20dark = v20.get('y0')
t_v20dark = v20.get('time')

plt.plot(t_v10 * 10, y0_v10, 'r', label='36km/h')
plt.plot(t_v15 * 15, y0_v15, 'g', label='54km/h')
plt.plot(t_v20 * 20, y0_v20, 'b', label='72km/h')

# plt.plot(t_v20*20, y0_v20, 'b', label='72km/h, bright')
# plt.plot(t_v20dark*20, y0_v20dark, 'k', label='72km/h, dark')

plt.xlim(200, 500)
plt.ylim(-1.75, 1.75)
plt.grid()
plt.legend()
plt.show()