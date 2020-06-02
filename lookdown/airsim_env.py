import numpy as np
from gym import Env
from gym.spaces import Box
import airsim
import time
from .midpts import MIDPTS #
from .midpts_curvature import MIDPTS_C #
import matplotlib.pyplot as plt
import cv2
from simple_pid import PID
import random
import pygame
from .graphic_tool import GraphicTool #
from .estimator.Estimator_CNN import Estimator_CNN #
from .controller_config import * #
import control

class AirsimEnv(Env):
    def __init__(self, carname):
        super(AirsimEnv, self).__init__()
        self.carname = carname
        self.action_space = Box(low=np.array([0.0]), high=np.array([1.0])) # [rho]
        self.observation_space = Box(low=np.array([0.0, 0.0, 0.0, 0.0]),
                                     high=np.array([np.inf, np.inf, np.inf, 1.0]))
        self.client = airsim.CarClient(ip="192.168.0.151")
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=carname)
        self.car_controls = airsim.CarControls()
        self.midpts = MIDPTS
        self.midpts_c = MIDPTS_C
        self.init_state = self.client.getCarState(carname)
        self.init_vel = self.init_state.kinematics_estimated.linear_velocity
        self.init_vel = np.array([self.init_vel.x_val, self.init_vel.y_val])
        self.init_position = self.init_state.kinematics_estimated.position
        self.init_orientation = self.init_state.kinematics_estimated.orientation
        self.init_pose = airsim.Pose(self.init_position, self.init_orientation)
        self.ref_speed = 10.0 # m/s
        self.client.simEnableWeather(True)
        self.pid_speed = PID(1.0, 0.1, 0.05, setpoint=self.ref_speed, output_limits=(-1.0, 1.0), sample_time=0.05)
        self.graphic = GraphicTool()
        self.estimator = Estimator_CNN()
        self.estimator.load_model('/home/myounghoe/stable-baselines/experiments/gain_scheduling/env/estimator/Models/')
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True, self.carname)
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        time.sleep(0.5)
        obs = self.make_observation()
        img = obs[5][80:][:, :, ::-1] # ROI, BGR to RGB,(80, 160, 3)
        img = np.expand_dims(img, axis=0) # (1, 80, 160, 3)
        ey_est, ey_unc, epsi_est, epsi_unc = self.estimator.get_estimation(img)
        Vx = obs[3]
        rho = 1.0
        self.current_Vx = obs[3]
        self.current_curvature = obs[6]
        self.current_ey_true = obs[1]
        self.current_epsi_true = obs[2]
        self.current_ey_est = ey_est
        self.current_epsi_est = epsi_est
        # self.current_R_est = R_est
        self.current_x_est = np.array([[self.current_ey_est], [0.0], [self.current_epsi_est], [0.0]])
        self.current_x_hat = np.array([[self.current_ey_est], [0.0], [self.current_epsi_est], [0.0]])
        self.current_x_true = np.array([[self.current_ey_true], [0.0], [self.current_epsi_true], [0.0]])
        self.current_delta = 0.0
        self.current_P = np.zeros((4, 4))
        self.timestep = 0
        self.recording_traj = []

        state = np.array([ey_unc, epsi_unc, Vx, rho])
        # state = np.array([ey_est, ey_unc, epsi_est, epsi_unc, Vx, 1.0])
        return state

    def step(self, action):
        # Apply action
        rho = action[0]
        rho = 1.0
        # if rho < 0.2:
        #     self.pid_speed.setpoint = (rho + 0.2) * self.ref_speed
        # else:
        #     self.pid_speed.setpoint = rho * self.ref_speed
        accel = self.pid_speed(self.current_Vx)
        if accel < 0.0:
            self.car_controls.throttle = 0.0
            self.car_controls.brake = abs(accel)
        elif accel >= 0.0:
            self.car_controls.throttle = accel
            self.car_controls.brake = 0.0
        delta = self.get_sf_control(self.current_x_est, self.current_curvature, self.current_Vx)
        # delta = self.get_sf_control(self.current_x_hat, self.current_curvature, self.current_Vx)
        scaled_delta = delta * rho
        self.car_controls.steering = scaled_delta
        self.client.setCarControls(self.car_controls, vehicle_name=self.carname)
        
        if self.timestep == 300:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 360:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        
        if self.timestep == 600:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 660:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        
        if self.timestep == 900:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 960:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        
        if self.timestep == 1200:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 1260:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
        
        if self.timestep == 1500:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 1560:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)

        if self.timestep == 1800:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 1860:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)

        if self.timestep == 2100:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)
        if self.timestep == 2160:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)

        obs = self.make_observation()
        img = obs[5][80:][:, :, ::-1] # ROI and BGR to RGB
        cv2.imshow('img', img)
        cv2.waitKey(1)
        ey_est, ey_unc, epsi_est, epsi_unc = self.estimator.get_estimation(np.expand_dims(img, axis=0))

        self.current_x_hat = self.KF(ey_est, epsi_est, ey_unc, epsi_unc, 20)

        ey_true = obs[1]
        epsi_true = obs[2]
        R_true = obs[6] # true curvature
        Vx = obs[3]
        car_pt = obs[4]

        ey_est_dot = (ey_est - self.current_ey_est) / 0.05
        epsi_est_dot = (epsi_est - self.current_epsi_est) / 0.05

        ey_true_dot = (ey_true - self.current_ey_true) / 0.05
        epsi_true_dot = (epsi_true - self.current_epsi_true) / 0.05
        
        self.graphic.update_plot(ey_true, ey_est, epsi_true, epsi_est, ey_unc, epsi_unc, rho)

        # ground truth values used for controller
        self.current_Vx = obs[3]
        self.current_curvature = obs[6]
        self.current_ey_true = obs[1]
        self.current_epsi_true = obs[2]
        self.current_R_true = obs[6]
        self.current_x_true = np.array([[ey_true], [ey_true_dot], [epsi_true], [epsi_true_dot]])

        # estimated values used for controller
        self.current_ey_est = ey_est
        self.current_epsi_est = epsi_est
        # self.current_R_est = R_est
        self.current_R_est = 0.001
        self.current_x_est = np.array([[ey_est], [ey_est_dot], [epsi_est], [epsi_est_dot]])
        self.current_delta = scaled_delta
        
        self.timestep += 1

        reward = self.compute_reward(obs)
        # state = np.array([ey_est, epsi_est, Vx, avg_unc])
        # state = np.array([ey_est, ey_unc, epsi_est, epsi_unc, Vx, rho])
        # state = np.array([ey_unc, epsi_unc, 0.0, Vx, action[0]])
        state = np.array([ey_unc, epsi_unc, Vx, rho])

        self.recording_traj.append(np.array([ey_unc, epsi_unc, rho, delta, scaled_delta, Vx, ey_est, self.current_x_hat[0][0], ey_true, epsi_est, self.current_x_hat[2][0], self.current_x_hat[1][0], self.current_x_hat[3][0]]))

        # Terminate or not
        done = False
        if abs(ey_true) > 1.8 or car_pt[0] > 600.0:
            done = True
            recording_traj = np.array(self.recording_traj, dtype=np.float16)
            np.savez_compressed('/home/myounghoe/stable-baselines/experiments/gain_scheduling/trajectory/0-1-0-1.npz', traj=recording_traj)
        return state, reward, done, {}

    def render(self):
        pass

    # Get all the necessary information
    def make_observation(self):
        car_state = self.client.getCarState(vehicle_name=self.carname)
        car_pt = car_state.kinematics_estimated.position
        car_pt = np.array([car_pt.x_val, car_pt.y_val, 0.0])
        car_vel = car_state.kinematics_estimated.linear_velocity
        car_vel = np.array([car_vel.x_val, car_vel.y_val])
        car_vel = car_vel - self.init_vel
        car_orientation = car_state.kinematics_estimated.orientation
        yaw = self.q_to_yaw(car_orientation)

        # Get nearest midpoint pair
        d_min = 100.0
        for i, item in enumerate(self.midpts):
            d = (car_pt[0] - item[0])**2 + (car_pt[1] - item[1])**2
            if d < d_min:
                d_min = d
                min_index = i
        curvature = self.midpts_c[min_index][2]
        nearest_pts = [np.append(self.midpts[min_index], 0.0), np.append(self.midpts[min_index+1], 0.0)]
        dist = np.linalg.norm(np.cross(car_pt - nearest_pts[0], car_pt - nearest_pts[1])) / np.linalg.norm(nearest_pts[0]-nearest_pts[1])
        dist = dist / 1.9
        if np.cross(car_pt - nearest_pts[0], car_pt - nearest_pts[1])[2] < 0.0:
            dist = -dist
        if dist < -1.0:
            dist += 2.0
        if dist > 1.0:
            dist -= 2.0        
        direction_vec = (nearest_pts[1] - nearest_pts[0])[:2]
        orientation_vec = np.array([np.cos(yaw), np.sin(yaw)])
        theta = np.arccos(np.dot(direction_vec, orientation_vec) / (np.linalg.norm(direction_vec)))
        if np.cross(np.append(direction_vec, 0), np.append(orientation_vec,0))[2] < 0.0:
            theta = -theta
        speed = car_state.speed
        while True:
            imgs = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
            if imgs[0].height != 0:
                break
        img = imgs[0]
        img = np.fromstring(img.image_data_uint8, dtype=np.uint8)
        img = img.reshape(imgs[0].height, imgs[0].width, 4)
        
        img = img[:,:,0:3] # image size becomes (height, width, 3)
        return np.array([car_vel, dist, theta, speed, car_pt, img, curvature])

    # jerk penalty using action variance
    def compute_reward(self, obs):
        v_ratio = obs[3] / self.ref_speed
        if v_ratio < 1.0:
            reward_prog = (np.cos(obs[2]) - abs(obs[1])) * v_ratio
        else:
            reward_prog = (np.cos(obs[2]) - abs(obs[1])) * (2*v_ratio - v_ratio**2)
        # steering_variance = np.var(cmd_history)
        # reward = reward_prog - 0.0 * steering_variance
        reward = reward_prog
        return reward

    def q_to_yaw(self, q):
        siny_cosp = +2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = +1.0 - 2.0 * (q.y_val**2 + q.z_val**2) 
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    # get state feedback control input
    # gain scheduling like LPV
    def get_sf_control(self, x, curvature, Vx):
        R = 1 / curvature
        if Vx < 20.0:
            K = K20
        if 20.0 <= Vx < 25.0:
            K = K20 + (K25-K20)*(Vx-20.0)/(25.0-20.0)
        if 25.0 <= Vx < 30.0:
            K = K25 + (K30-K25)*(Vx-25.0)/(30.0-25.0)
        if 30.0 <= Vx:
            K = K30
        delta = -np.matmul(K, x)[0]
        delta_ff = m*Vx**2/(R*L)*(lr/(2*Cf)-lf/(2*Cr)+lf/(2*Cr)*K[2])+L/R-lr/R*K[2]
        steering = delta + delta_ff
        return steering

    def KF(self, ey_est, epsi_est, ey_unc, epsi_unc, Vx):
        F_k, B_k = self.get_sysmat(Vx)  # (4, 4), (4, 1)
        z_k = np.array([[ey_est],       # (2, 1)
                        [epsi_est]])
        R_k = np.array([[ey_unc, 0],    # (2, 2)
                        [0, epsi_unc]])
        x_k0 = np.matmul(F_k, self.current_x_hat) + B_k * self.current_delta
        P_k0 = np.matmul(np.matmul(F_k, self.current_P), np.transpose(F_k)) + Q_k
        S_k = np.matmul(np.matmul(H_k, P_k0), np.transpose(H_k)) + R_k
        K_k = np.matmul(np.matmul(P_k0, np.transpose(H_k)), np.linalg.inv(S_k)) # (4, 2)
        x_k = x_k0 + np.matmul(K_k, z_k - np.matmul(H_k, x_k0))
        P_k = np.matmul((np.eye(4) - np.matmul(K_k, H_k)), P_k0)
        x_k = np.array(x_k)
        P_k = np.array(P_k)
        self.current_P = P_k
        self.current_x_hat = x_k
        return x_k

    def get_sysmat(self, Vx):
        A = np.array([[0, 1, 0, 0],
                      [0, -(2*Cf+2*Cr)/(m*Vx), (2*Cf+2*Cr)/m, (-2*Cf*lf+2*Cr*lr)/(m*Vx)],
                      [0, 0, 0, 1],
                      [0, -(2*Cf*lf-2*Cr*lr)/(Iz*Vx), (2*Cf*lf-2*Cr*lr)/Iz, -(2*Cf*lf**2+2*Cr*lr**2)/(Iz*Vx)]])
        B = np.array([[0],
                      [2*Cf/m],
                      [0],
                      [2*Cf*lf/Iz]])
        C = np.eye(4)
        D = np.zeros((4, 1))
        sys = control.StateSpace(A, B, C, D)
        sysd = sys.sample(0.05)
        F = sysd.A
        B = sysd.B
        return F, B