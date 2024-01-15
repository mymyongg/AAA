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
import random
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

class CarlaEnv:
    def __init__(self, sun_altitude_angle=90.0):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        weather = carla.WeatherParameters(cloudiness=0.0, # [0., 100.]
                                          precipitation=0.0, # [0., 100.]
                                          precipitation_deposits=0.0, # [0., 100.]
                                          wind_intensity=0.0, # [0., 100.]
                                          fog_density=0.0, # [0., 100.]
                                          fog_distance=0.0, # [0., inf]
                                          wetness=0.0, # [0., 100.]
                                          sun_azimuth_angle=0.0, # [0., 360.]
                                          sun_altitude_angle=sun_altitude_angle) # [-90., 90.]
        self.world.set_weather(weather)

        self.top_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.top_camera_transform = carla.Transform(carla.Location(x=100.0, y=-100.0, z=120.0), carla.Rotation(pitch=-90.0, yaw=-90.0))
        self.top_camera_bp.set_attribute('image_size_x', '1080')
        self.top_camera_bp.set_attribute('image_size_y', '1080')
        self.top_camera_bp.set_attribute('sensor_tick', '0.5')
        self.top_camera_actor = self.world.spawn_actor(self.top_camera_bp, self.top_camera_transform)
        self.top_camera_actor.listen(lambda image: self.get_top_image(image))

        self.back_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.back_camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        self.back_camera_bp.set_attribute('image_size_x', '720')
        self.back_camera_bp.set_attribute('image_size_y', '480')
        self.back_camera_bp.set_attribute('sensor_tick', '0.1')
        self.back_camera_actor = self.world.spawn_actor(self.back_camera_bp, self.back_camera_transform, attach_to=self.ego_vehicle_actor)
        self.back_camera_actor.listen(lambda image: self.get_back_image(image))
        
        time.sleep(1)
        
    def get_top_image(self, image):
        self.current_top_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))
    def get_back_image(self, image):
        self.current_back_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))