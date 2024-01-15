import csv
import env
import os
import time
import cv2
import numpy as np

env = env.CarlaEnv() # -90.0 to 90.0
settings = env.world.get_settings()
settings.fixed_delta_seconds = 0.05
settings.synchronous_mode = True
env.world.apply_settings(settings)

dataset_path = 'system_id'
f_name = 'data_throttle_75.csv'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

f = open(os.path.join(dataset_path, f_name), 'w', newline='')
wr = csv.writer(f)

total_frame = 0

for t in [5, 4, 3, 2, 1]: # Count down 5 sec before collection starts
    print('Collection starts in {} sec'.format(t))
    time.sleep(1)
print('Start!')

while True:
    env.world.tick()
    obs = env.make_observation()
    # [img, [L1, y_L1, eps_L1, K_L1], [L2, y_L2, eps_L2, K_L2], ..., [L8, y_L8, eps_L8, K_L8], v_x, v_y, yaw_rate, steer, throttle, brake]

    y_0 = obs[1][1]
    eps_0 = obs[1][2]
    v_x = obs[-6]
    v_y = obs[-5]
    yaw_rate = obs[-4]
    steer = obs[-3]

    wr.writerow([v_y, yaw_rate, y_0, eps_0, steer, v_x])

    total_frame += 1
    if total_frame % 100 == 0:
        print('{} frames collected.'.format(total_frame))
    if total_frame == 1000:
        break

print('Collection finished!')

f.close()