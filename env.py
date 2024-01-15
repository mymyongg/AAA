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
        self.L_list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        
        self.front_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.front_camera_transform = carla.Transform(carla.Location(x=0.8, z=1.6))
        self.front_camera_bp.set_attribute('image_size_x', '320')
        self.front_camera_bp.set_attribute('image_size_y', '180')
        self.front_camera_bp.set_attribute('sensor_tick', '0.01')
        self.front_camera_actor = self.world.spawn_actor(self.front_camera_bp, self.front_camera_transform, attach_to=self.ego_vehicle_actor)

        self.back_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.back_camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        self.back_camera_bp.set_attribute('image_size_x', '320')
        self.back_camera_bp.set_attribute('image_size_y', '180')
        self.back_camera_bp.set_attribute('sensor_tick', '0.1')
        self.back_camera_actor = self.world.spawn_actor(self.back_camera_bp, self.back_camera_transform, attach_to=self.ego_vehicle_actor)

        self.top_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.top_camera_transform = carla.Transform(carla.Location(x=100.0, y=-100.0, z=120.0), carla.Rotation(pitch=-90.0, yaw=-90.0))
        self.top_camera_bp.set_attribute('image_size_x', '100')
        self.top_camera_bp.set_attribute('image_size_y', '100')
        self.top_camera_bp.set_attribute('sensor_tick', '0.5')
        self.top_camera_actor = self.world.spawn_actor(self.top_camera_bp, self.top_camera_transform)
        
        self.current_front_image = None
        self.front_camera_actor.listen(lambda image: self.get_front_image(image))
        self.back_camera_actor.listen(lambda image: self.get_back_image(image))
        self.top_camera_actor.listen(lambda image: self.get_top_image(image))
        
        time.sleep(1)
        # cv2.namedWindow('front camera', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('warped image', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('top camera', cv2.WINDOW_NORMAL)

    def get_front_image(self, image):
        self.current_front_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))

    def get_back_image(self, image):
        self.current_back_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))
        
    def get_top_image(self, image):
        self.current_top_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape((image.height, image.width, -1))

    # def warp_image(self, src):
    #     srcPoint = np.array([[290, 200], [350, 200], [640, 360], [0, 360]], dtype=np.float32)
    #     dstPoint = np.array([[0, 0], [640, 0], [640, 360], [0, 360]], dtype=np.float32)
    #     matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    #     dst = cv2.warpPerspective(src, matrix, (640, 360))
    #     dst = cv2.resize(dst, (80, 360), interpolation=cv2.INTER_AREA)
    #     return dst
    
    def destroy(self):
        actors = [self.ego_vehicle_actor,
                self.front_camera_actor]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def get_waypoints_map(self):
        spawn_points = self.map.get_spawn_points()
        waypoints = []
        for spawn_point in spawn_points:
            current_wp = self.map.get_waypoint(spawn_point.location)
            wps = current_wp.next_until_lane_end(0.1)
            for wp in wps:
                location = wp.transform.location
                rotation = wp.transform.rotation
                forward_vector = rotation.get_forward_vector()
                waypoints.append({'x':location.x, 'y':location.y, 'forward_vector':np.array([forward_vector.x, forward_vector.y]), 's':wp.s})

        return waypoints

    def make_observation(self):
        obs = {}
        
        img = self.current_front_image[90:, :, :3] # (90, 320, 3) BGR. If want BGR to RGB, add [..., ::-1].
        obs['img'] = img
        
        yaw_rate = -self.ego_vehicle_actor.get_angular_velocity().z * np.pi / 180.0 # deg/sec to rad/sec
        obs['yaw_rate'] = yaw_rate
        
        # Calculating vx and vy
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
        obs['vx'] = vx
        obs['vy'] = vy

        nearest_wp = self.map.get_waypoint(vehicle_transform.location)
        obs['waypoint_s'] = nearest_wp.s
        
        wps_ahead = []
        info_all = []
        for d in np.arange(0.1, 15.0, 0.1):
            wps_ahead.append(nearest_wp.next(d)[0])
        obs['waypoints_ahead'] = wps_ahead
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

            info_d = [L, y_L, eps_L, K_L]
            info_all.append(info_d)
        info_all = np.array(info_all)

        for i in self.L_list:
            obs['L'+str(int(i))] = info_all[np.argmin(np.absolute(info_all[:, 0] - i))]

        control = self.ego_vehicle_actor.get_control()
        steer = -control.steer * 70.0 * np.pi / 180.0 # Max 70 deg to rad
        throttle = control.throttle
        brake = control.brake
        obs['steer'] = steer
        obs['throttle'] = throttle
        obs['brake'] = brake

        location = vehicle_transform.location
        rotation = vehicle_transform.rotation
        abs_forward_vector = vehicle_transform.get_forward_vector()
        abs_velocity = self.ego_vehicle_actor.get_velocity()
        obs['abs_location'] = np.array([location.x, location.y])
        obs['abs_yaw'] = rotation.yaw * np.pi / 180.0
        obs['abs_forward_vector'] = np.array([abs_forward_vector.x, abs_forward_vector.y])
        obs['abs_velocity'] = np.array([abs_velocity.x, abs_velocity.y])

        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        obs['timestamp'] = timestamp
        
        return obs # img, L0, L2, L4, L6, L8, L10, vx, vy, yaw_rate, steer, throttle, brake, abs_location, abs_yaw, abs_forward_vector, abs_velocity, timestamp