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
from simple_pid import PID
from env import CarlaEnv
import cv2
import time
import numpy as np
import pygame
from graphic_tool import GraphicTool
import matplotlib.pyplot as plt
import control
from constants import *
import matlab.engine
import keyboard

def kalman(yL_pred, epsL_pred, yL_unc, epsL_unc, vx, L, x_00, u, P_00):
    Q = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    H = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1]])
    F, B = get_sysmat(vx, L)  # (4, 4), (4, 1)
    z = np.array([[yL_pred],       # (2, 1)
                    [epsL_pred]])
    # R = np.array([[yL_unc, 0],    # (2, 2)
    #               [0, epsL_unc]])
    R = np.array([[1, 0],    # (2, 2)
                    [0, 1]])
    x_10 = np.matmul(F, x_00) + B * u
    P_10 = np.matmul(np.matmul(F, P_00), np.transpose(F)) + Q
    S = np.matmul(np.matmul(H, P_10), np.transpose(H)) + R
    K = np.matmul(np.matmul(P_10, np.transpose(H)), np.linalg.inv(S)) # (4, 2)
    x_11 = x_10 + np.matmul(K, z - np.matmul(H, x_10))
    P_11 = np.matmul((np.eye(4) - np.matmul(K, H)), P_10)

    x_11 = np.array(x_11)
    P_11 = np.array(P_11)
    return x_11, P_11

def get_sysmat(vx, L):
    A = np.array([[-(c_f+c_r)/(m*vx), -vx+(c_r*l_r-c_f*l_f)/(m*vx), 0, 0],
                  [(-l_f*c_f+l_r*c_r)/(I_psi*vx), -(l_f**2*c_f+l_r**2*c_r)/(I_psi*vx), 0, 0],
                  [-1, -L, 0, vx],
                  [0, -1, 0, 0]])
    B = np.array([[c_f/m],
                  [(l_f*c_f)/I_psi],
                  [0],
                  [0]])
    C = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    D = np.array([[0],
                  [0],
                  [0]])

    sys = control.StateSpace(A, B, C, D)
    sysd = sys.sample(0.05)
    F = sysd.A
    B = sysd.B
    return F, B

def lowpass(tau, ts, pre_y, x):
    y = (tau * pre_y + ts * x) / (tau + ts)
    return y

def main():
    # Setting
    sun_altitude = 30.0
    sampling_time = 0.05
    img_size = (90, 320, 3)
    lookahead_list = [2, 4, 6, 8, 10, 20, 30]
    num_pred = 2 * len(lookahead_list)
    num_mixture = 3
    model_path = 'successful_model/3lane_M14_mixed'
    lpf_tau = 0.01
    lookahead = 6
    ref_speed = 15

    env = CarlaEnv(sun_altitude)
    settings = env.world.get_settings()
    settings.fixed_delta_seconds = sampling_time
    settings.synchronous_mode = True
    env.world.apply_settings(settings)
    model = CMDN(img_size=img_size, M=num_pred, KMIX=num_mixture)
    model.load_model(model_path)
    pid_controller = PID(1.0, 0.1, 0.05, setpoint=ref_speed, output_limits=(0.0, 1.0), sample_time=sampling_time) # ref speed
    graphic = GraphicTool()
    eng = matlab.engine.start_matlab()

    # Initial values
    x_pred_kf = np.zeros((4, 1))
    P_kf = np.zeros((4, 4))
    u = 0.0
    yL_pred_lpf = 0.0
    epsL_pred_lpf = 0.0
    xc_matlab = matlab.double([0.0, 0.0, 0.0, 0.0])
    
    # Placeholder
    gt = []
    pred = []
    pred_kf = []
    pred_lpf = []
    e_true = {}
    e_pred = {}
    
    try:
        while True:
            env.world.tick()

            if keyboard.is_pressed('u'):
                lookahead += 2
                ref_speed += 5
                pid_controller.setpoint = ref_speed
                print('L:{}, V:{}'.format(lookahead, ref_speed))
            if keyboard.is_pressed('d'):
                lookahead -= 2
                ref_speed -= 5
                pid_controller.setpoint = ref_speed
                print('L:{}, V:{}'.format(lookahead, ref_speed))

            obs = env.make_observation() # [img, [L1, y_L1, eps_L1, K_L1], [L2, y_L2, eps_L2, K_L2], ..., [L8, y_L8, eps_L8, K_L8], vx, vy, yaw_rate, steer, throttle, brake]
            img = np.array([obs[0] / 255.0])
            vx = obs[-6]
            vy = obs[-5]
            psi_dot = obs[-4]
            
            total_expectation, uncertainty = model.get_estimation(img)
            total_expectation = total_expectation.numpy()[0] # (M,)
            uncertainty = uncertainty.numpy()[0] # (M, M)

            for i in range(len(lookahead_list)):
                e_pred['y'+str(lookahead_list[i])+'_pred'] = total_expectation[i]
                e_pred['eps'+str(lookahead_list[i])+'_pred'] = total_expectation[i+len(lookahead_list)]
                e_pred['y'+str(lookahead_list[i])+'_unc'] = uncertainty[i, i]
                e_pred['eps'+str(lookahead_list[i])+'_unc'] = uncertainty[i+len(lookahead_list), i+len(lookahead_list)]
            
            # modulator = np.clip(-5.0 * y8_unc + 1.0, 0.5, 1.0)
            # pid_controller.setpoint = ref_speed * modulator
            # lookahead = ref_speed * modulator * 0.4

            if 0 <= lookahead < 1.0:
                yL_pred = e_pred['y0_pred']
                yL_unc = e_pred['y0_unc']
                epsL_pred = e_pred['eps0_pred']
                epsL_unc = e_pred['eps0_unc']
            if 1.0 <= lookahead < 3.0:
                yL_pred = e_pred['y2_pred']
                yL_unc = e_pred['y2_unc']
                epsL_pred = e_pred['eps2_pred']
                epsL_unc = e_pred['eps2_unc']
            if 3.0 <= lookahead < 5.0:
                yL_pred = e_pred['y4_pred']
                yL_unc = e_pred['y4_unc']
                epsL_pred = e_pred['eps4_pred']
                epsL_unc = e_pred['eps4_unc']
            if 5.0 <= lookahead < 7.0:
                yL_pred = e_pred['y6_pred']
                yL_unc = e_pred['y6_unc']
                epsL_pred = e_pred['eps6_pred']
                epsL_unc = e_pred['eps6_unc']
            if 7.0 <= lookahead < 9.0:
                yL_pred = e_pred['y8_pred']
                yL_unc = e_pred['y8_unc']
                epsL_pred = e_pred['eps8_pred']
                epsL_unc = e_pred['eps8_unc']
            
            # CMDN prediction
            x_pred = np.array([[vy], [psi_dot], [yL_pred], [epsL_pred]])
            x_pred_matlab = matlab.double([vy, psi_dot, yL_pred, epsL_pred])

            # Kalman filter
            # x_pred_kf, P_kf = kalman(yL_pred, epsL_pred, yL_unc, epsL_unc, ref_speed, lookahead, x_pred_kf, u, P_kf)
            # yL_pred_kf = x_pred_kf[2][0]
            # epsL_pred_kf = x_pred_kf[3][0]

            # Low pass filter
            # yL_pred_lpf = lowpass(tau=lpf_tau, ts=sampling_time, pre_y=yL_pred_lpf, x=yL_pred)
            # epsL_pred_lpf = lowpass(tau=lpf_tau, ts=sampling_time, pre_y=epsL_pred_lpf, x=epsL_pred)
            # x_pred_lpf = np.array([[vy],
            #                        [psi_dot],
            #                        [yL_pred_lpf],
            #                        [epsL_pred_lpf]])

            # Ground truth
            for i in range(len(lookahead_list)):
                e_true['y'+str(lookahead_list[i])+'_true'] = obs[i+1][1]
                e_true['eps'+str(lookahead_list[i])+'_true'] = obs[i+1][2]

            yL_true = e_true['y'+str(lookahead)+'_true']
            epsL_true = e_true['eps'+str(lookahead)+'_true']
           
            x_true = np.array([[vy],
                               [psi_dot],
                               [yL_true],
                               [epsL_true]])

            # For plotting kf
            gt.append(yL_true)
            pred.append(yL_pred)
            # pred_kf.append(yL_pred_kf)
            # pred_lpf.append(yL_pred_lpf)

            graphic.update_plot(yL_true, yL_pred, epsL_true, epsL_pred, e_pred['y10_unc'], e_pred['y20_unc'], e_pred['y30_unc'])

            # print('y10:{:10.2f}, eps10:{:10.2f}, vx:{:10.2f}, vy:{:10.2f}, yaw_rate:{:10.2f}, steer:{:10.2f}, K:{:10.4f}'.format(y10, eps10, vx, vy, yaw_rate, steer, K_10))

            K = optimal_gain['V'+str(int(ref_speed))+'_L'+str(int(lookahead))]

            # u_ff = K_10 * (l - ((l_f * c_f - l_r * c_r) * vx**2 * m) / (c_r * c_f * l)) # Feed forward control
            # u = -np.matmul(K, x_true)[0]
            # u = -np.matmul(K, x_pred)[0]            
            # u = -np.matmul(K, x_pred_lpf)[0]

            u, xc_matlab = eng.cal_u_dynamic(float(ref_speed), x_pred_matlab, xc_matlab, nargout=2)

            throttle = pid_controller(vx)
            control = carla.VehicleControl(throttle=throttle, steer=-u)
            env.ego_vehicle_actor.apply_control(control)

    finally:
        settings.synchronous_mode = False
        env.world.apply_settings(settings)

        plt.plot(gt, 'b', label='ground truth')
        plt.plot(pred, 'r', label='prediction')
        # plt.plot(pred_kf, 'g', label='kalman filter')
        # plt.plot(pred_lpf, 'k', label='low pass filter')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Bye!')
        pass