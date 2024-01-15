import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

ours = np.load('plot_data/ours.npz')

time = ours.get('timestamp')
vel = ours.get('speed')
y4 = ours.get('y4_true')
y6 = ours.get('y6_true')
y8 = ours.get('y8_true')
y10 = ours.get('y10_true')
eps4 = ours.get('eps4_true')
eps6 = ours.get('eps6_true')
eps8 = ours.get('eps8_true')
eps10 = ours.get('eps10_true')
unc_y4 = ours.get('unc_y4')
unc_y6 = ours.get('unc_y6')
unc_y8 = ours.get('unc_y8')
unc_y10 = ours.get('unc_y10')
unc_eps4 = ours.get('unc_eps4')
unc_eps6 = ours.get('unc_eps6')
unc_eps8 = ours.get('unc_eps8')
unc_eps10 = ours.get('unc_eps10')
K4 = ours.get('K4_true')
K6 = ours.get('K6_true')
K8 = ours.get('K8_true')
K10 = ours.get('K10_true')
steer = ours.get('steer')
lookahead = ours.get('lookahead')

f_name = 'joohwan.csv'

f = open(f_name, 'w', newline='')
wr = csv.writer(f)

for i in range(time.shape[0]):
    wr.writerow([time[i], vel[i], y4[i], y6[i], y8[i], y10[i], eps4[i], eps6[i], eps8[i], eps10[i], unc_y4[i], unc_y6[i], unc_y8[i], unc_y10[i], unc_eps4[i], unc_eps6[i], unc_eps8[i], unc_eps10[i], K4[i], K6[i], K8[i], K10[i], steer[i], lookahead[i]])
    
f.close()