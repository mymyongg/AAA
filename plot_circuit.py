from __future__ import print_function
import gi
gi.require_version('Gtk', '2.0')

import glob
import os
import sys

try:
    sys.path.append(glob.glob('/home/mymyongg/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import env
import matplotlib.pyplot as plt

env = env.CarlaEnv()
vehicle_transform = env.ego_vehicle_actor.get_transform()
nearest_wp = env.map.get_waypoint(vehicle_transform.location)
wps = nearest_wp.next_until_lane_end(0.1)
x = []
y = []
s = []
print(len(wps))
for wp in wps:
    x.append(wp.transform.location.x)
    y.append(wp.transform.location.y)
    s.append(wp.s)

# plt.plot(x[:1000], y[:1000], label='fog level 0')
# plt.plot(x[1000:2000], y[1000:2000], label='fog level 1')
# plt.plot(x[2000:3000], y[2000:3000], label='fog level 2')
# plt.plot(x[3000:4000], y[3000:4000], label='fog level 1')
# plt.plot(x[4000:5000], y[4000:5000], label='fog level 0')
# plt.plot(x[5000:6000], y[5000:6000], label='fog level 1')
# plt.plot(x[6000:], y[6000:], label='fog level 2')

plt.plot(x[:1000]+x[4000:5000], y[:1000]+y[4000:5000], 'r.', label='fog level 0')
plt.plot(x[1000:2000]+x[3000:4000]+x[5000:6000], y[1000:2000]+y[3000:4000]+y[5000:6000], 'b.', label='fog level 1')
plt.plot(x[2000:3000]+x[6000:], y[2000:3000]+y[6000:], 'g.', label='fog level 2')
plt.axes().set_aspect('equal')
# plt.axis('off')
plt.legend()
# plt.plot(x, y, 'r--')
plt.show()