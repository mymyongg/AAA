from __future__ import print_function
from CMDN import CMDN
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

from env_playback import CarlaEnv
import time
import cv2

image_path = 'record_image/ours'

env = CarlaEnv()
env.client.set_replayer_time_factor(1.0)
env.client.replay_file('/home/mymyongg/carla_log/ours.log', 0.1, 60.0, 0)

for i in range(100):
    imagename = os.path.join(image_path, '{}.png'.format(i))
    cv2.imwrite(imagename, env.current_back_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    time.sleep(0.1)