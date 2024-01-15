from CMDN import CMDN
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

model_path = 'successful_model/fog_024'
image_path = 'dataset/images'
label_path = 'dataset/labels'

image_folders = os.listdir(image_path)
image_folders.sort()

label_list = os.listdir(label_path)
label_list.sort()

# Load image data
dataset_images = []   
for folder in image_folders:
    file_list = os.listdir(os.path.join(image_path, folder))
    file_list.sort()
    for png_file in file_list:
        png = cv2.imread(os.path.join(image_path, folder, png_file))
        dataset_images.append(png)
dataset_images = np.array(dataset_images)

# Load label data
dataset_labels = []
for label_file in label_list:
    labels = np.load(os.path.join(label_path, label_file))
    labels = labels.get('labels')
    labels = np.concatenate([labels[:, 1:, 1], labels[:, 1:, 2]], axis=1) # (N, 16)
    dataset_labels.append(labels)
dataset_labels = np.concatenate(dataset_labels, axis=0)

# Generate test set
# Fog level 0
x_test = dataset_images[:5000][2500:3000]
y_test = dataset_labels[:5000][2500:3000]
# Fog level 1
# x_test = dataset_images[5000:10000][2500:3000]
# y_test = dataset_labels[5000:10000][2500:3000]
# Fog level 2
# x_test = dataset_images[10000:][2500:3000]
# y_test = dataset_labels[10000:][2500:3000]

x_test = x_test / 255.0

# Load model
estimator = CMDN(img_size=(90, 320, 3), M=8, KMIX=3)
estimator.load_model(model_path)

# Predict
y_pred = estimator.CMDN_model.predict(x_test)
pi, mu, sigma = estimator.get_GMM_params(y_pred)
total_expectation, uncertainty = estimator.get_estimation(x_test) # uncertainty: (N, M, M)
pred = np.array(total_expectation) # (N, M)

yL_pred = pred[:, :4]
epsL_pred = pred[:, 4:]

unc = []
for i in range(uncertainty.shape[1]):
    unc.append(uncertainty[:, i, i])
unc = np.array(unc) # (M, N)

# Plot result
yL_limit = max(abs(y_test[:, :4]).max(), abs(yL_pred).max()) * 1.1
epsL_limit = max(abs(y_test[:, 4:]).max(), abs(epsL_pred).max()) * 1.1
sigma_yL_limit = unc[:4, :].max() * 1.1
sigma_epsL_limit = unc[4:, :].max() * 1.1

yL_limit = 2.6
epsL_limit = 0.4
sigma_yL_limit = 1.6
sigma_epsL_limit = 0.04


time = np.linspace(0.05, 25, 500)

fig = plt.figure(figsize=(25, 8))
# plt.rc('font', size=10)
# plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=13)
# plt.rc('xtick', labelsize=)
# plt.rc('ytick', labelsize=)
plt.rc('legend', fontsize=12)
# plt.rc('figure', titlesize=BIGGER_SIZE)

for i, L in enumerate([4, 6, 8, 10]):
    plt.subplot(4, 4, i+1)
    if i==0:
        plt.plot(time, y_test[:, i], 'b-', label='true')
        plt.plot(time, yL_pred[:, i], 'r:', label='prediction')
        plt.legend(loc=3)
    else:
        plt.plot(time, y_test[:, i], 'b-')
        plt.plot(time, yL_pred[:, i], 'r:')
    plt.ylim(-yL_limit, yL_limit)
    plt.ylabel('$y_{'+str(L)+'}$ [m]')
    plt.grid(True)

    plt.subplot(4, 4, i+5)
    plt.plot(time, unc[i, :], 'k')
    plt.ylim(-0.05, sigma_yL_limit)
    plt.ylabel('$\sigma_{y_{'+str(L)+'}}$ [m]')
    plt.grid(True)

    plt.subplot(4, 4, i+9)
    if i==0:
        plt.plot(time, y_test[:, i+4], 'b-', label='true')
        plt.plot(time, epsL_pred[:, i], 'r:', label='prediction')
        plt.legend(loc=3)
    else:
        plt.plot(time, y_test[:, i+4], 'b-')
        plt.plot(time, epsL_pred[:, i], 'r:')
    plt.ylim(-epsL_limit, epsL_limit)
    plt.ylabel('$\epsilon_{'+str(L)+'}$ [rad]')
    plt.grid(True)

    plt.subplot(4, 4, i+13)
    plt.plot(time, unc[i+4, :], 'k')
    plt.ylim(-0.001,sigma_epsL_limit)
    plt.xlabel('time [s]')
    plt.ylabel('$\sigma_{\epsilon_{'+str(L)+'}}$ [rad]')
    plt.grid(True)

plt.show()