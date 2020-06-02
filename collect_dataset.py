import env
import os
import time
import cv2
import numpy as np
from simple_pid import PID

sun_altitude_angle = -10.0

env = env.CarlaEnv(sun_altitude_angle) # -90.0 to 90.0
settings = env.world.get_settings()
settings.fixed_delta_seconds = 0.05
settings.synchronous_mode = True
env.world.apply_settings(settings)

image_path = 'dataset/images/{}'.format(int(sun_altitude_angle))
# image_path = 'dataset/test_images/test_{}'.format(int(sun_altitude_angle))

if not os.path.exists(image_path):
    os.makedirs(image_path)

label_path = 'dataset/labels'
# label_path = 'dataset/test_labels'

if not os.path.exists(label_path):
    os.makedirs(label_path)

total_frame = 0

labels = []

for t in [5, 4, 3, 2, 1]: # Count down 5 sec before collection starts
    print('Collection starts in {} sec'.format(t))
    time.sleep(1)
print('Start!')

while True:
    env.world.tick()
    observation = env.make_observation()
    image = observation[0] 
    # label = observation[1:11] # [[L1, y_L1, eps_L1], [L2, y_L2, eps_L2], ..., [L8, y_L8, eps_L8]] -> Shape is (8, 3).
    label = observation[2:7] + [observation[8]] + [observation[10]]
    labels.append(label) # Shape is (T, 8, 3)
    imagename = os.path.join(image_path, '{0:05d}.png'.format(total_frame))
    cv2.imwrite(imagename, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    total_frame += 1
    if total_frame % 100 == 0:
        print('{} frames collected.'.format(total_frame))
    if total_frame == 20000:
        break

settings.synchronous_mode = False
env.world.apply_settings(settings)

labels = np.array(labels, dtype=np.float16)
print('Collection finished!')

filename = os.path.join(label_path, '{}.npz'.format(int(sun_altitude_angle)))
np.savez_compressed(filename, labels=labels)