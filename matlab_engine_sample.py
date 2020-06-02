import matlab.engine
from matlab import double as double_m

from ctypes import *
# import pandas
import csv
import numpy as np
import math
import time

from sys import platform as _platform
import os

if not os.path.exists('./Tests'):
    os.makedirs('./Tests')

class FullcarBenchmark:

    def __init__(self):
        
        CDLL._func_restype_ = c_float
        if _platform == "linux" or _platform == "linux2":
            # linux
            print("No dll for linux")
            exit(0)
        elif _platform == "darwin":
            # MAC OS X
            self.dll = CDLL("Fullcar_model_osx.dylib")
            print("MAC dll")
        else:
            self.dll = CDLL("Fullcar_model_win64.dll")
            print("Windows dll")

        disable_matlab = False
        if disable_matlab == False:
            self.matlab_eng = None
            self._init_matlab()
        self.dll.Fullcar_model_initialize()
        self.dll.restype = c_float

        self.episode_cnt = 0
        self.step_cnt = 0
        self.matlab_cnt = 0.0

        self.Cmax = 4000
        self.Cmin = 300
        self.R = 0.1
        self.state_X = np.zeros((14,1))
        self.count = 0

        self.state_recon = True

        self.state_SH = [0,0,0,0,0,0,0,0]
    
    def read_csv(self,log_dir):
        dataset = np.genfromtxt(log_dir + '.csv', delimiter = ',')
        self.filename = log_dir
        self.road_r_total = dataset[1:,-1]
        self.road_l_total = dataset[1:,-2]
        self.car_vel_total = dataset[1:,-3]

    def _init_matlab(self):
        if self.matlab_eng == None:
            # Launch Matlab if not running
            self.matlab_eng = matlab.engine.start_matlab()
            self.matlab_eng.desktop(nargout=0)

            print("Matlab Loaded")

        self._reset_matlab(self.matlab_eng)
    
    def _reset_matlab(self, eng):
        self.dll.Fullcar_model_terminate()
        self.dll.Fullcar_model_initialize()

        self.resetKKFVar()

        eng.Road_Generator_Fullcar(1.0,0.0)

        # self.road_time = eng.eval('t')
        # self.road_zl = eng.eval('road_ZL')
        # self.road_zr = eng.eval('road_ZR')
        # self.car_vel = eng.eval('car_vel')
        # self.road_index = eng.eval('Road_index')
        self.Kx = eng.eval('Kx')
        # print(self.Kx)

    def _assignWorkspace(self, eng, var_str, var):
        eng.workspace[var_str] = var

    def _terminate(self):
        self.matlab_eng.quit()

    def skyhook_calculator(self,upper_vel,delta_vel):
        Cmax = 4000
        Cmin = 300
        epsilon = 0.0001
        alpha = 0.5
        sat_limit = 1000
        if upper_vel * delta_vel >= 0:
            C = (alpha * Cmax * upper_vel + (1 - alpha) * Cmax * upper_vel)/(delta_vel + epsilon)
            C = min(C,Cmax)
            u = C * delta_vel
        else:
            u = Cmin*delta_vel
        
        if u >= 0:
            if u > sat_limit:
                u_ = sat_limit
            else:
                u_ = u
        else:
            if u < -sat_limit:
                u_ = -sat_limit
            else:
                u_ = u
        return u_
    
    def passive(self):
        dz_fl = self.state_SH[0]
        dz_fr = self.state_SH[2]
        dz_rl = self.state_SH[4]
        dz_rr = self.state_SH[6]

        vel_fl = dz_fl - self.state_SH[1]
        vel_fr = dz_fr - self.state_SH[3]
        vel_rl = dz_rl - self.state_SH[5]
        vel_rr = dz_rr - self.state_SH[7]

        u_fl = 1000 * vel_fl
        u_fr = 1000 * vel_fr
        u_rl = 1000 * vel_rl
        u_rr = 1000 * vel_rr

        return [u_fl,u_fr,u_rl,u_rr]

    def skyhook(self):
                 
        # self.state_SH = [fl2,tfl2,fr2,tfr2,rl2,trl2,rr2,trr2]
        dz_fl = self.state_SH[0]
        dz_fr = self.state_SH[2]
        dz_rl = self.state_SH[4]
        dz_rr = self.state_SH[6]

        vel_fl = dz_fl - self.state_SH[1]
        vel_fr = dz_fr - self.state_SH[3]
        vel_rl = dz_rl - self.state_SH[5]
        vel_rr = dz_rr - self.state_SH[7]

        u_fl = self.skyhook_calculator(dz_fl,vel_fl)
        u_fr = self.skyhook_calculator(dz_fr,vel_fr)
        u_rl = self.skyhook_calculator(dz_rl,vel_rl)
        u_rr = self.skyhook_calculator(dz_rr,vel_rr)
        
        return [u_fl,u_fr,u_rl,u_rr]

    def iteration(self):
        for method in ['lqr','sh','pas']:
            for i in range(50):
                self._reset_matlab(self.matlab_eng)
                self.count = i+1
                self.road_zl = self.road_l_total[i*2000:((i+1)*2000)]
                self.road_zr = self.road_r_total[i*2000:((i+1)*2000)]
                self.car_vel = self.car_vel_total[i*2000:((i+1)*2000)]
                self.main(method)
                print(method, self.count)

    def resetKKFVar(self):
        self.A_bp_body = np.array([[1.9822,-0.9824,0,0],
                            [1,0,0,0],
                            [0.7101, -0.7101, 0.7633, -0.2814],
                            [0,0,1,0]])
        self.B_bp_body = np.array([[0.3582],[0],[0.1283],[0]])
        self.C_bp_body = np.array([0.7101,-0.7101,0.7633,-1.2814])
        self.D_bp_body = 0.1283

        self.A_bp_tire = np.array([[1.8937, -0.8995, 0, 0],
                                   [1, 0, 0, 0],
                                   [0.9861, -0.9891, 0.0429, -0.1923],
                                   [0, 0, 1, 0]])
        self.B_bp_tire = np.array([[0.5207],[0],[0.2711],[0]])
        self.C_bp_tire = np.array([0.9861, -0.9891, 0.0429, -1.1923])
        self.D_bp_tire = 0.2711

        self.filtered_fl = 0; self.dummy_fl = np.array([[0],[0],[0],[0]])
        self.filtered_fr = 0; self.dummy_fr = np.array([[0],[0],[0],[0]])
        self.filtered_rl = 0; self.dummy_rl = np.array([[0],[0],[0],[0]])
        self.filtered_rr = 0; self.dummy_rr = np.array([[0],[0],[0],[0]])
        self.filtered_tfl = 0; self.dummy_tfl = np.array([[0],[0],[0],[0]])
        self.filtered_tfr = 0; self.dummy_tfr = np.array([[0],[0],[0],[0]])
        self.filtered_trl = 0; self.dummy_trl = np.array([[0],[0],[0],[0]])
        self.filtered_trr = 0; self.dummy_trr = np.array([[0],[0],[0],[0]])

        self.P_tfl = np.zeros([2,2])
        self.P_tfr = np.zeros([2,2])
        self.P_trl = np.zeros([2,2])
        self.P_trr = np.zeros([2,2])
        self.P_fl = np.zeros([2,2])
        self.P_fr = np.zeros([2,2])
        self.P_rl = np.zeros([2,2])
        self.P_rr = np.zeros([2,2])

        self.hx_tfl = np.zeros([2,1])
        self.hx_tfr = np.zeros([2,1])
        self.hx_trl = np.zeros([2,1])
        self.hx_trr = np.zeros([2,1])
        self.hx_fl = np.zeros([2,1])
        self.hx_fr = np.zeros([2,1])
        self.hx_rl = np.zeros([2,1])
        self.hx_rr = np.zeros([2,1])

        self.W_body = 100
        self.V_body = 0.01

        self.W_tire = 100
        self.V_tire = 0.01

    def kkf(self,in_,position):
        T = 0.01
        Phi = np.array([[1,T],
                        [0,1]])
        Q = np.array([[T**3/3,T**2/2],
                      [T**2/2,T]])
        C = np.array([[0,1]])

        if position == 'fl':
            P = self.P_fl; hx = self.hx_fl
            acc = self.C_bp_body @ self.dummy_fl + self.D_bp_body * in_
            self.dummy_fl = self.A_bp_body @ self.dummy_fl + self.B_bp_body * in_
            W = self.W_body; V = self.V_body        
        elif position == 'fr':
            P = self.P_fr; hx = self.hx_fr
            acc = self.C_bp_body @ self.dummy_fr + self.D_bp_body * in_
            self.dummy_fr = self.A_bp_body @ self.dummy_fr + self.B_bp_body * in_
            W = self.W_body; V = self.V_body        
        elif position == 'rl':
            P = self.P_rl; hx = self.hx_rl
            acc = self.C_bp_body @ self.dummy_rl + self.D_bp_body * in_
            self.dummy_rl = self.A_bp_body @ self.dummy_rl + self.B_bp_body * in_
            W = self.W_body; V = self.V_body        
        elif position == 'rr':
            P = self.P_rr; hx = self.hx_rr
            acc = self.C_bp_body @ self.dummy_rr + self.D_bp_body * in_
            self.dummy_rr = self.A_bp_body @ self.dummy_rr + self.B_bp_body * in_
            W = self.W_body; V = self.V_body
        elif position == 'tfl':
            P = self.P_tfl; hx = self.hx_tfl
            acc = self.C_bp_tire @ self.dummy_tfl + self.D_bp_tire * in_
            self.dummy_tfl = self.A_bp_tire @ self.dummy_tfl + self.B_bp_tire * in_
            W = self.W_tire; V = self.V_tire
        elif position == 'tfr':
            P = self.P_tfr; hx = self.hx_tfr
            acc = self.C_bp_tire @ self.dummy_tfr + self.D_bp_tire * in_
            self.dummy_tfr = self.A_bp_tire @ self.dummy_tfr + self.B_bp_tire * in_
            W = self.W_tire; V = self.V_tire
        elif position == 'trl':
            P = self.P_trl; hx = self.hx_trl
            acc = self.C_bp_tire @ self.dummy_trl + self.D_bp_tire * in_
            self.dummy_trl = self.A_bp_tire @ self.dummy_trl + self.B_bp_tire * in_
            W = self.W_tire; V = self.V_tire
        elif position == 'trr':
            P = self.P_trr; hx = self.hx_trr
            acc = self.C_bp_tire @ self.dummy_trr + self.D_bp_tire * in_
            self.dummy_trr = self.A_bp_tire @ self.dummy_trr + self.B_bp_tire * in_
            W = self.W_tire; V = self.V_tire  
        
        K = P @ np.transpose(C) * 1 / (C @ P @ np.transpose(C) + V)
        xk = hx + K @ (acc - C @ hx)
        p_buff = (np.eye(2) - K @ C) @ P
        hx = Phi @ xk
        P = Phi @ p_buff * np.transpose(Phi) + Q * W

        if position == 'fl':
            self.P_fl = P
            self.hx_fl = hx
        elif position == 'fr':
            self.P_fr = P
            self.hx_fr = hx
        elif position == 'rl':
            self.P_rl = P
            self.hx_rl = hx
        elif position == 'rr':
            self.P_rr = P
            self.hx_rr = hx
        elif position == 'tfl':
            self.P_tfl = P
            self.hx_tfl = hx
        elif position == 'tfr':
            self.P_tfr = P
            self.hx_tfr = hx
        elif position == 'trl':
            self.P_trl = P
            self.hx_trl = hx
        elif position == 'trr':
            self.P_trr = P
            self.hx_trr = hx
        
        return xk

    def main(self, method):
        u_fl = 0
        u_fr = 0
        u_rl = 0
        u_rr = 0
        # print(self.road_zl.shape)
        # print(self.road_zl)
        road_FL = self.road_zl[0]
        road_FR = self.road_zr[0]
        vel_L = self.car_vel[0]
        vel_R = self.car_vel[0]

        for i in range(10000) :
            self.dll.setRoad_FL(c_float(road_FL))
            self.dll.setRoad_FR(c_float(road_FR))
            self.dll.setVel_L(c_float(vel_L))
            self.dll.setVel_R(c_float(vel_R))

            self.dll.setU_FL(c_float(u_fl))
            self.dll.setU_FR(c_float(u_fr))
            self.dll.setU_RL(c_float(u_rl))
            self.dll.setU_RR(c_float(u_rr))

            self.dll.Fullcar_model_step()

            if i % 5 == 0:
                k = i//5
                road_FL = self.road_zl[k]
                road_FR = self.road_zr[k]
                vel_L = self.car_vel[k]
                vel_R = self.car_vel[k]

                fl1 = self.dll.getStateFL1()
                fl2 = self.dll.getStateFL2()
                fl3 = self.dll.getStateFL3()
                rl1 = self.dll.getStateRL1()
                rl2 = self.dll.getStateRL2()
                rl3 = self.dll.getStateRL3()
                fr1 = self.dll.getStateFR1()
                fr2 = self.dll.getStateFR2()
                fr3 = self.dll.getStateFR3()
                rr1 = self.dll.getStateRR1()
                rr2 = self.dll.getStateRR2()
                rr3 = self.dll.getStateRR3()

                tfl1 = self.dll.getStateTFL1()
                tfl2 = self.dll.getStateTFL2()
                tfl3 = self.dll.getStateTFL3()

                trl1 = self.dll.getStateTRL1()
                trl2 = self.dll.getStateTRL2()
                trl3 = self.dll.getStateTRL3()

                tfr1 = self.dll.getStateTFR1()
                tfr2 = self.dll.getStateTFR2()
                tfr3 = self.dll.getStateTFR3()

                trr1 = self.dll.getStateTRR1()
                trr2 = self.dll.getStateTRR2()
                trr3 = self.dll.getStateTRR3()

                z1 = self.dll.getStateZ1()
                z2 = self.dll.getStateZ2()
                z3 = self.dll.getStateZ3()

                ddphi = self.dll.getStatePhiAcc()
                dphi = self.dll.getStatePhi1()
                phi = self.dll.getStatePhi2()

                ddtheta = self.dll.getStateThetaAcc()
                dtheta = self.dll.getStateTheta1()
                theta = self.dll.getStateTheta2()
                
                if self.state_recon == True:
                    [[fl_kk2],[trash]] = self.kkf(fl1,'fl')
                    [[rl_kk2],[trash]] = self.kkf(rl1,'rl')
                    [[fr_kk2],[trash]] = self.kkf(fr1,'fr')
                    [[rr_kk2],[trash]] = self.kkf(rr1,'rr')

                    [[tfl_kk2],[trash]] = self.kkf(tfl1,'tfl')
                    [[trl_kk2],[trash]] = self.kkf(trl1,'trl')
                    [[tfr_kk2],[trash]] = self.kkf(tfr1,'tfr')
                    [[trr_kk2],[trash]] = self.kkf(trr1,'trr')
                    self.state_SH = [fl_kk2,tfl_kk2,fr_kk2,tfr_kk2,rl_kk2,trl_kk2,rr_kk2,trr_kk2]
                else :
                    self.state_SH = [fl2,tfl2,fr2,tfr2,rl2,trl2,rr2,trr2]

                state = [fl1, fl2, fl3, fr1, fr2, fr3, rl1, rl2, rl3, rr1, rr2, rr3, tfl1, tfl2, tfl3, tfr1, tfr2, tfr3, trl1, trl2, trl3, trr1, trr2, trr3, z1, z2, z3, ddphi, dphi, phi, ddtheta, dtheta, theta]
                state_lqr = [z3-0.4, z2, phi, dphi, theta, dtheta, tfl3-0.3, tfl2, tfr3-0.3, tfr2, trl3-0.3, trl2, trr3-0.3, trr2]

                if method == 'sh':
                    [u_fl, u_fr, u_rl, u_rr] = self.skyhook()
                elif method == 'pas':
                    [u_fl, u_fr, u_rl, u_rr] = self.passive()
                elif method == 'lqr':
                    [u_fl, u_fr, u_rl, u_rr] = -np.matmul(self.Kx, np.transpose(state_lqr))
                    
                    # u_ = u.tolist()
                    # [u_fl, u_fr, u_rl, u_rr] = u_
                    

                row = state + [u_fl, u_fr, u_rl, u_rr]

                if os.path.isfile('rough_road_kkf_' + method +'.csv'):
                    with open('rough_road_kkf_' + method +'.csv','a') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(row)
                else:
                    with open('rough_road_kkf_' + method +'.csv','w') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(['fl1','fl2','fl3','fr1','fr2','fr3','rl1','rl2','rl3','rr1','rr2','rr3','tfl1','tfl2','tfl3','tfr1','tfr2','tfr3','trl1','trl2','trl3','trr1','trr2','trr3','z1','z2','z3','ddphi','dphi','phi','ddtheta','dtheta','theta','u_FL','u_FR','u_RL','u_RR'])
                        writer.writerow(row)
                
                # with open('Tests/sh_fullroad_6.csv','a') as fd:
                #     row = state + [u_fl, u_fr, u_rl, u_rr] + [road_FL, road_FR, vel_L]
                #     writer = csv.writer(fd)
                #     writer.writerow(row)
    

if __name__=='__main__':
    fullcar = FullcarBenchmark()
    # fullcar.iteration()
    logName = "sac_fc_benchmark_2x64_window2"
    fullcar.read_csv(logName)
    fullcar.iteration()

