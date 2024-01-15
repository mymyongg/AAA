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
import numpy as np
import pygame
from graphic_tool import GraphicTool
import matplotlib.pyplot as plt
import control
from constants import *
import matlab.engine
import time

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
    # Set parameters
    sun_altitude = 30.0
    sampling_time = 0.02
    img_size = (90, 320, 3)
    lookahead_list = [4, 6, 8, 10]
    num_pred = 2 * len(lookahead_list)
    num_mixture = 3
    model_path = 'successful_model/fog_024'
    # lpf_tau = 0.01
    unc_window = 5
    lookahead = 8
    ref_speed = 20
    constant_speed = True
    v_min = 10

    env = CarlaEnv(sun_altitude)
    for t in range(10):
        print('Change settings in {} sec'.format(t))
        time.sleep(1)

    settings = env.world.get_settings()
    settings.fixed_delta_seconds = sampling_time
    settings.synchronous_mode = True
    env.world.apply_settings(settings)
    model = CMDN(img_size=img_size, M=num_pred, KMIX=num_mixture)
    model.load_model(model_path)
    pid_controller = PID(1.0, 0.1, 0.05, setpoint=ref_speed, output_limits=(-1.0, 1.0), sample_time=sampling_time)
    graphic = GraphicTool()
    pygame.init()
    print("Starting Matlab...")
    eng = matlab.engine.start_matlab()
    print('Matlab is started!')

    # Initial values
    # x_pred_kf = np.zeros((4, 1))
    # P_kf = np.zeros((4, 4))
    u = 0.0
    # yL_pred_lpf = 0.0
    # epsL_pred_lpf = 0.0
    xc_matlab = matlab.double([0.0, 0.0, 0.0, 0.0])
    timestamp = 0
    ckpt_100 = ckpt_200 = ckpt_300 = ckpt_400 = ckpt_500 = ckpt_600 = ckpt_700 = ckpt_800 = ckpt_900 = False

    # Placeholders
    true = {}
    pred = {}
    unc_list = {}
    avg_unc = {}
    plot_value = {}
    for L in lookahead_list:
        plot_value['y'+str(L)+'_pred'] = []
        plot_value['eps'+str(L)+'_pred'] = []
        plot_value['y'+str(L)+'_true'] = []
        plot_value['eps'+str(L)+'_true'] = []
        plot_value['K'+str(L)+'_true'] = []
        plot_value['unc_y'+str(L)] = []
        plot_value['unc_eps'+str(L)] = []
        plot_value['avg_unc_y'+str(L)] = []
        plot_value['avg_unc_eps'+str(L)] = []
        unc_list['y'+str(L)] = [0.0] * unc_window
        unc_list['eps'+str(L)] = [0.0] * unc_window
    plot_value['y0_true'] = []
    plot_value['eps0_true'] = []
    plot_value['lookahead'] = []
    plot_value['ref_speed'] = []
    plot_value['speed'] = []
    plot_value['timestamp'] = []
    plot_value['distance'] = []
    plot_value['steer'] = []
    
    # pred_kf = []
    # pred_lpf = []
    
    try:
        while True:
            env.world.tick()

            # Keyboard control of scheduling parameters
            # for event in pygame.event.get():
            #     if event.type == pygame.KEYUP:
            #         if event.key == pygame.K_u:
            #             lookahead += 2
            #             ref_speed += 5
            #             pid_controller.setpoint = ref_speed
            #             print('L:{}, V:{}'.format(lookahead, ref_speed))
            #         if event.key == pygame.K_d:
            #             lookahead -= 2
            #             ref_speed -= 5
            #             pid_controller.setpoint = ref_speed
            #             print('L:{}, V:{}'.format(lookahead, ref_speed))

            obs = env.make_observation()
            img = np.array([obs['img'] / 255.0])
            vx = obs['vx']
            vy = obs['vy']
            psi_dot = obs['yaw_rate']
            
            # CMDN prediction
            total_expectation, uncertainty = model.get_estimation(img)
            total_expectation = total_expectation.numpy()[0] # (M,)
            uncertainty = uncertainty.numpy()[0] # (M, M)

            for i, L in enumerate(lookahead_list):
                pred['y'+str(L)] = total_expectation[i]
                pred['eps'+str(L)] = total_expectation[i+len(lookahead_list)]
                pred['y'+str(L)+'_unc'] = uncertainty[i, i]
                pred['eps'+str(L)+'_unc'] = uncertainty[i+len(lookahead_list), i+len(lookahead_list)]

                del unc_list['y'+str(L)][0]
                unc_list['y'+str(L)].append(uncertainty[i, i])
                avg_unc['y'+str(L)] = np.mean(unc_list['y'+str(L)])
                del unc_list['eps'+str(L)][0]
                unc_list['eps'+str(L)].append(uncertainty[i+len(lookahead_list), i+len(lookahead_list)])
                avg_unc['eps'+str(L)] = np.mean(unc_list['eps'+str(L)])

            # Ground truth
            for i, L in enumerate(lookahead_list):
                true['y'+str(L)] = obs['L'+str(L)][1]
                true['eps'+str(L)] = obs['L'+str(L)][2]

            # Choose schduling parameters from uncertainty
            if constant_speed == False:
                lookahead = 4
                ref_speed = 10
                while avg_unc['y'+str(lookahead)] < 0.01:
                    lookahead += 2
                    ref_speed += 5
                    if lookahead == 8:
                        break
                pid_controller.setpoint = ref_speed

            # State from prediction
            yL_pred = pred['y'+str(lookahead)]
            yL_unc = pred['y'+str(lookahead)+'_unc']
            epsL_pred = pred['eps'+str(lookahead)]
            epsL_unc = pred['eps'+str(lookahead)+'_unc']
            
            x_pred = np.array([[vy], [psi_dot], [yL_pred], [epsL_pred]])
            x_pred_matlab = matlab.double([vy, psi_dot, yL_pred, epsL_pred])
            
            # State from ground truth
            yL_true = true['y'+str(lookahead)]
            epsL_true = true['eps'+str(lookahead)]

            x_true = np.array([[vy],
                               [psi_dot],
                               [yL_true],
                               [epsL_true]])
            x_true_matlab = matlab.double([vy, psi_dot, yL_true, epsL_true])

            # For plotting
            for L in lookahead_list:
                plot_value['y'+str(L)+'_pred'].append(pred['y'+str(L)])
                plot_value['eps'+str(L)+'_pred'].append(pred['eps'+str(L)])
                plot_value['y'+str(L)+'_true'].append(obs['L'+str(L)][1])
                plot_value['eps'+str(L)+'_true'].append(obs['L'+str(L)][2])
                plot_value['K'+str(L)+'_true'].append(obs['L'+str(L)][3])
                plot_value['unc_y'+str(L)].append(pred['y'+str(L)+'_unc'])
                plot_value['unc_eps'+str(L)].append(pred['eps'+str(L)+'_unc'])
                plot_value['avg_unc_y'+str(L)].append(np.mean(unc_list['y'+str(L)]))
                plot_value['avg_unc_eps'+str(L)].append(np.mean(unc_list['eps'+str(L)]))
        
            plot_value['y0_true'].append(obs['L0'][1])
            plot_value['eps0_true'].append(obs['L0'][2])
            plot_value['lookahead'].append(lookahead)
            plot_value['speed'].append(vx)
            timestamp += sampling_time
            plot_value['timestamp'].append(timestamp)
            plot_value['distance'].append(obs['waypoint_s'])
            plot_value['steer'].append(obs['steer'])

            # Graphical visualization
            graphic.update_plot(yL_true, yL_pred, epsL_true, epsL_pred, pred['y4_unc'], pred['y6_unc'], pred['y8_unc'], pred['y10_unc'])

            # Feedforward control
            # u_ff = K_10 * (l - ((l_f * c_f - l_r * c_r) * vx**2 * m) / (c_r * c_f * l))
            
            # LPV control from Matlab
            u, xc_matlab = eng.cal_u_dynamic(float(ref_speed), x_pred_matlab, xc_matlab, nargout=2)
            # u, xc_matlab = eng.cal_u_dynamic(float(ref_speed), x_true_matlab, xc_matlab, nargout=2)

            accel = pid_controller(vx)
            if accel < 0.0:
                throttle = 0.0
                brake = abs(accel)
            else:
                throttle = accel
                brake = 0.0

            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=-u)
            env.ego_vehicle_actor.apply_control(control)

            if timestamp > 5 and ckpt_100 == False:
                ckpt_100 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 10 and ckpt_200 == False:
                ckpt_200 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 15 and ckpt_300 == False:
                ckpt_300 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 20 and ckpt_400 == False:
                ckpt_400 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 25 and ckpt_500 == False:
                ckpt_500 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 30 and ckpt_600 == False:
                ckpt_600 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 35 and ckpt_700 == False:
                ckpt_700 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 40 and ckpt_800 == False:
                ckpt_800 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            if timestamp > 45 and ckpt_900 == False:
                ckpt_900 = True
                for t in range(10):
                    print('Change fog settings in {} sec'.format(t))
                    time.sleep(1)
            
            if timestamp > 30 or abs(obs['L0'][1]) > 1.70:
                break


    finally:
        settings.synchronous_mode = False
        env.world.apply_settings(settings)

        # np.savez('plot_data/v'+str(ref_speed)+'.npz', y0_true=plot_value['y0_true'],
        np.savez('plot_data_RL/20mps.npz',  y0_true=plot_value['y0_true'],
                                        eps0_true=plot_value['eps0_true'],
                                        y4_true=plot_value['y4_true'],
                                        eps4_true=plot_value['eps4_true'],
                                        K4_true=plot_value['K4_true'],
                                        y6_true=plot_value['y6_true'],
                                        eps6_true=plot_value['eps6_true'],
                                        K6_true=plot_value['K6_true'],
                                        y8_true=plot_value['y8_true'],
                                        eps8_true=plot_value['eps8_true'],
                                        K8_true=plot_value['K8_true'],
                                        y10_true=plot_value['y10_true'],
                                        eps10_true=plot_value['eps10_true'],
                                        K10_true=plot_value['K10_true'],
                                        unc_y4=plot_value['unc_y4'],
                                        unc_y6=plot_value['unc_y6'],
                                        unc_y8=plot_value['unc_y8'],
                                        unc_y10=plot_value['unc_y10'],
                                        unc_eps4=plot_value['unc_eps4'],
                                        unc_eps6=plot_value['unc_eps6'],
                                        unc_eps8=plot_value['unc_eps8'],
                                        unc_eps10=plot_value['unc_eps10'],
                                        avg_unc_y4=plot_value['avg_unc_y4'],
                                        avg_unc_y6=plot_value['avg_unc_y6'],
                                        avg_unc_y8=plot_value['avg_unc_y8'],
                                        avg_unc_y10=plot_value['avg_unc_y10'],
                                        avg_unc_eps4=plot_value['avg_unc_eps4'],
                                        avg_unc_eps6=plot_value['avg_unc_eps6'],
                                        avg_unc_eps8=plot_value['avg_unc_eps8'],
                                        avg_unc_eps10=plot_value['avg_unc_eps10'],
                                        lookahead=plot_value['lookahead'],
                                        speed=plot_value['speed'],
                                        steer=plot_value['steer'],
                                        distance=plot_value['distance'],
                                        timestamp=plot_value['timestamp'])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Bye!')
        pass