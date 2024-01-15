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

def main():
    env = CarlaEnv()
    settings = env.world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    env.world.apply_settings(settings)
    cv2.namedWindow('top camera', cv2.WINDOW_NORMAL)

    try:
        while True:
            env.world.tick()
            cv2.imshow('top camera', env.current_top_image)
            cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()
        settings.synchronous_mode = False
        env.world.apply_settings(settings)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Bye!')
        pass