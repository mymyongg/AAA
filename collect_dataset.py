import env
import os
import time
import cv2
import numpy as np
from simple_pid import PID
from graphic_tool import GraphicTool

sun_altitude_angle = 30
fog_density = 1112
sampling_time = 0.02
image_path = 'dataset/images/fog_{}'.format(int(fog_density))
label_path = 'dataset/labels'
if not os.path.exists(image_path):
    os.makedirs(image_path)
if not os.path.exists(label_path):
    os.makedirs(label_path)

env = env.CarlaEnv(sun_altitude_angle) # -90.0 to 90.0
graphic = GraphicTool()
for t in range(20):
    print('Change settings in {} sec'.format(t))
    time.sleep(1)

settings = env.world.get_settings()
settings.fixed_delta_seconds = sampling_time
settings.synchronous_mode = True
env.world.apply_settings(settings)

total_frame = 0
labels = []

for t in [5, 4, 3, 2, 1]: # Count down 5 sec before collection starts
    print('Collection starts in {} sec'.format(t))
    time.sleep(1)
print('Start!')

while True:
    env.world.tick()
    obs = env.make_observation()
    graphic.draw_map(obs)
    image = obs['img']
    label = np.stack([obs['L2'], obs['L4'], obs['L6'], obs['L8'], obs['L10']], axis=0)
    labels.append(label) # [N, L, 3]
    imagename = os.path.join(image_path, '{0:05d}.png'.format(total_frame))
    cv2.imwrite(imagename, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    total_frame += 1
    if total_frame % 500 == 0:
        print('{} frames collected.'.format(total_frame))
    if total_frame == 5000:
        break

settings.synchronous_mode = False
env.world.apply_settings(settings)

labels = np.stack(labels, axis=0)
filename = os.path.join(label_path, 'fog_{}.npz'.format(int(fog_density)))
np.savez_compressed(filename, labels=labels)