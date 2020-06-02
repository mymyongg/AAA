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
  def __init__(self, sun_altitude_angle=30.0):
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
    self.ego_vehicle_actor = self.world.get_actors().filter('vehicle.*')[0]
    # self.ego_vehicle_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    # self.ego_vehicle_actor = self.world.try_spawn_actor(self.ego_vehicle_bp, random.choice(self.map.get_spawn_points()))
    
    self.front_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
    self.front_camera_transform = carla.Transform(carla.Location(x=0.8, z=1.6))
    self.front_camera_bp.set_attribute('image_size_x', '320')
    self.front_camera_bp.set_attribute('image_size_y', '180')
    self.front_camera_bp.set_attribute('sensor_tick', '0.1')
    self.front_camera_actor = self.world.spawn_actor(self.front_camera_bp, self.front_camera_transform, attach_to=self.ego_vehicle_actor)

    # self.back_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
    # self.back_camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
    # self.back_camera_bp.set_attribute('image_size_x', '320')
    # self.back_camera_bp.set_attribute('image_size_y', '180')
    # self.back_camera_bp.set_attribute('sensor_tick', '0.1')
    # self.back_camera_actor = self.world.spawn_actor(self.back_camera_bp, self.back_camera_transform, attach_to=self.ego_vehicle_actor)

    # self.top_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
    # self.top_camera_transform = carla.Transform(carla.Location(x=8.0, z=20.0), carla.Rotation(pitch=-90.0))
    # self.top_camera_bp.set_attribute('image_size_x', '640')
    # self.top_camera_bp.set_attribute('image_size_y', '360')
    # self.top_camera_bp.set_attribute('sensor_tick', '0.1')
    # self.top_camera_actor = self.world.spawn_actor(self.top_camera_bp, self.top_camera_transform, attach_to=self.ego_vehicle_actor)
    
    self.current_image = None
    self.front_camera_actor.listen(lambda image: self.get_front_image(image))
    # self.back_camera_actor.listen(lambda image: self.get_back_image(image))
    # self.top_camera_actor.listen(lambda image: self.get_top_image(image))
    time.sleep(1)
    cv2.namedWindow('front camera', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('warped image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('top camera', cv2.WINDOW_NORMAL)

  def get_front_image(self, image):
    # BGRA (H, W, C)
    self.current_front_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))

  def get_back_image(self, image):
    # BGRA (H, W, C)
    self.current_back_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))
    
  def get_top_image(self, image):
    # Raw data is array of BGRA 32-bit pixels
    self.current_top_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))

  def warp_image(self, src):
    srcPoint = np.array([[290, 200], [350, 200], [640, 360], [0, 360]], dtype=np.float32)
    dstPoint = np.array([[0, 0], [640, 0], [640, 360], [0, 360]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    dst = cv2.warpPerspective(src, matrix, (640, 360))
    dst = cv2.resize(dst, (80, 360), interpolation=cv2.INTER_AREA)
    return dst
  
  def destroy(self):
    actors = [self.ego_vehicle_actor,
              self.front_camera_actor]
    for actor in actors:
      if actor is not None:
        actor.destroy()

  def make_observation(self):
    observation = []
    
    img = self.current_front_image[90:, :, :3] # (90, 320, 3) BGR. If want BGR to RGB, add [..., ::-1].
    observation.append(img)
    
    yaw_rate = -self.ego_vehicle_actor.get_angular_velocity().z * np.pi / 180.0 # deg/sec to rad/sec
    
    vehicle_transform = self.ego_vehicle_actor.get_transform()
    
    v_vector = self.ego_vehicle_actor.get_velocity()
    v_vector = [v_vector.x, v_vector.y, 0.0]
    f_vector = vehicle_transform.get_forward_vector()
    f_vector = [f_vector.x, f_vector.y, 0.0]
    f_cross_v = np.cross(f_vector, v_vector)
    angle_fv = np.arcsin(np.clip(np.linalg.norm(f_cross_v) / np.linalg.norm(f_vector) / np.linalg.norm(v_vector), -1.0, 1.0))
    if f_cross_v[2] > 0.0:
        angle_fv = -angle_fv
    vx = np.linalg.norm(v_vector) * np.cos(angle_fv)
    vy = np.linalg.norm(v_vector) * np.sin(angle_fv)
    if np.isnan(vx) or np.isnan(vy):
      vx = 0.0
      vy = 0.0

    nearest_wp = self.map.get_waypoint(vehicle_transform.location)
    wps_ahead = []
    # for i in [0.1, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]:
    for i in [0.01, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
      wps_ahead.append(nearest_wp.next(i)[0])
    for wp in wps_ahead:
      PW = wp.transform.location.__sub__(vehicle_transform.location) # Vector from vehicle position to waypoint
      PW = [PW.x, PW.y, 0.0]
      wp_f_vector = wp.transform.get_forward_vector()
      wp_f_vector = [wp_f_vector.x, wp_f_vector.y, 0.0]
      theta = np.arccos(np.clip(np.dot(f_vector, PW) / np.linalg.norm(f_vector) / np.linalg.norm(PW), -1.0, 1.0))
      L = np.linalg.norm(PW) * np.cos(theta)
      y_L = np.linalg.norm(PW) * np.sin(theta)
      if np.cross(f_vector, PW)[2] > 0.0:
        y_L = -y_L
      eps_L = np.arccos(np.clip(np.dot(f_vector, wp_f_vector), -1.0, 1.0) / np.linalg.norm(f_vector) / np.linalg.norm(wp_f_vector))
      if np.cross(f_vector, wp_f_vector)[2] > 0.0:
        eps_L = -eps_L

      wp2 = wp.next(0.1)[0]
      wp2_f_vector = wp2.transform.get_forward_vector()
      wp2_f_vector = [wp2_f_vector.x, wp2_f_vector.y, 0.0]
      wpf_cross_wp2f = np.cross(wp_f_vector, wp2_f_vector)
      d_alpha = np.arcsin(np.clip(np.linalg.norm(wpf_cross_wp2f) / np.linalg.norm(wp_f_vector) / np.linalg.norm(wp2_f_vector), -1.0, 1.0))
      if wpf_cross_wp2f[2] > 0.0:
        d_alpha = -d_alpha
      K_L = d_alpha / (wp2.s - wp.s)

      observation_L = [L, y_L, eps_L, K_L]
      observation.append(observation_L)

    control = self.ego_vehicle_actor.get_control()
    steer = -control.steer * 70.0 * np.pi / 180.0 # Max 70 deg to rad
    throttle = control.throttle
    brake = control.brake
    
    observation.append(vx)
    observation.append(vy)
    observation.append(yaw_rate)
    observation.append(steer)
    observation.append(throttle)
    observation.append(brake)
    
    return observation # [img, [L1, y_L1, eps_L1, K_L1], [L2, y_L2, eps_L2, K_L2], ..., [L10, y_L10, eps_L10, K_L10], v_x, v_y, yaw_rate, steer, throttle, brake]