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

from env import CarlaEnv
import cv2
import time
import numpy as np
import pygame

# Parameters
m = 2090.0
I_psi = 2500.0
l_f = 1.0056
l_r = 2.875 - l_f
c_f = 154371.0
c_r = c_f
l = 2.875

V15_L10_R1 = np.array([0.0599, 0.2641, -0.3595, -0.7320])
V15_L10_R10 = np.array([0.0428, 0.1452, -0.2120, -0.5451])
V20_L20_R10 = np.array([0.0409, 0.2136, -0.1787, -0.4182])

def main():
    env = CarlaEnv(30)
    settings = env.world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    env.world.apply_settings(settings)

    while True:
        env.world.tick()
        obs = env.make_observation()
        # [img, [L1, y_L1, eps_L1, K_L1], [L2, y_L2, eps_L2, K_L2], ..., [L8, y_L8, eps_L8, K_L8], v_x, v_y, yaw_rate, steer, throttle, brake]
        y_0 = obs[1][1]
        eps_0 = obs[1][2]
        y_10 = obs[3][1]
        eps_10 = obs[3][2]
        y_20 = obs[5][1]
        eps_20 = obs[5][2]
        v_x = obs[-6]
        v_y = obs[-5]
        yaw_rate = obs[-4]
        steer = obs[-3]
        K_10 = -obs[3][3]
        print('y_10:{:10.2f}, eps_10:{:10.2f}, v_x:{:10.2f}, v_y:{:10.2f}, yaw_rate:{:10.2f}, steer:{:10.2f}, K:{:10.4f}'.format(y_10, eps_10, v_x, v_y, yaw_rate, steer, K_10))
        x = np.array([[v_y],
                      [yaw_rate],
                      [y_20],
                      [eps_20]])
        # u_ff = K_10 * (l - ((l_f * c_f - l_r * c_r) * v_x**2 * m) / (c_r * c_f * l))
        K = V20_L20_R10
        u = -np.matmul(K, x)[0]
        if np.isnan(u):
            u = 0.0
        control = carla.VehicleControl(throttle=0.72, steer=-u)
        env.ego_vehicle_actor.apply_control(control)


if __name__ == '__main__':
    main()