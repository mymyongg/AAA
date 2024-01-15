import pickle
import matplotlib.pyplot as plt
import os

with open('dataset_lim/waypoints.pkl', 'rb') as f:
    wps = pickle.load(f)
    w_x = []
    w_y = []
    for wp in wps:
        w_x.append(wp['x'])
        w_y.append(wp['y'])
    plt.plot(w_x, w_y, label='center line')

data_list = os.listdir('dataset_lim/conservative')
data_list.sort()

for data in data_list:
    with open(os.path.join('dataset_lim/conservative', data), 'rb') as f:
        traj = pickle.load(f)
        x = []
        y = []
        for point in traj:
            x.append(point['abs_location'][0])
            y.append(point['abs_location'][1])
        plt.plot(x, y)

plt.legend()
plt.show()