import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

reward = []

for e in tf.train.summary_iterator('/home/mymyongg/uncertainty_aware_project/stable-baselines/experiments/gain_scheduling/ppo2_tensorboard/rl_steering/events.out.tfevents.1567742389.myounghoe'):
    for v in e.summary.value:
        if v.tag == 'episode_reward' or v.tag == 'accuracy':
            reward.append(v.simple_value)

reward = np.array(reward) * 10
timestep = np.arange(len(reward)) * 10

df = pd.DataFrame(reward)


fig = plt.figure(figsize=(3.34, 1.6))

plt.rc('font', size=7)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=7)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=7)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.plot(timestep, df[0], 'lightblue')
plt.plot(timestep, df[0].rolling(15).mean(), 'b:')
plt.xlabel('Training step')
plt.ylabel('Episode reward')
plt.grid(True)
plt.show()
# plt.plot(df[])
# smooth = []








# plt.plot(timestep, reward, 'k-')
# plt.ylabel('Episode reward')
# plt.xlabel('Training step')
# plt.grid(True)
# plt.show()






# def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value

#     return smoothed