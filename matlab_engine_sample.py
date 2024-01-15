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

