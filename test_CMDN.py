from CMDN import CMDN
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# image_path = 'dataset/test_images'
# label_path = 'dataset/test_labels'
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
    labels = np.concatenate([labels[:, 0:7, 1], labels[:, 0:7, 2]], axis=1) # (N, 16)
    dataset_labels.append(labels)
dataset_labels = np.concatenate(dataset_labels, axis=0)

x_test = dataset_images[100:]
y_test = dataset_labels[100:, :]

x_test = x_test / 255.0

# Load trained model
estimator = CMDN(img_size=(90, 320, 3), M=14, KMIX=3)
model_path = 'successful_model/3lane_M14'
estimator.load_model(model_path)

# Predict
y_pred = estimator.CMDN_model.predict(x_test)
pi, mu, sigma = estimator.get_GMM_params(y_pred)
total_expectation, uncertainty = estimator.get_estimation(x_test) # uncertainty: (N, M, M)

y_L_pred = total_expectation # (N, M)
y_L_pred = np.array(y_L_pred)

unc = []
for i in range(uncertainty.shape[1]):
    unc.append(uncertainty[:, i, i])
unc = np.array(unc) # (M, N)

# Plot result
y_test_max = y_test.max()
y_pred_max = y_L_pred.max()
y_max = max(y_test_max, y_pred_max) + 2.0

y_test_min = y_test.min()
y_pred_min = y_L_pred.min()
y_min = min(y_test_min, y_pred_min) - 2.0

row = 2
column = 7

for i in range(row * column):
    axis = plt.subplot(row, column, i+1)
    plt.plot(y_test[:, i], 'b', label='ground truth')
    plt.plot(y_L_pred[:, i], 'r', label='prediction')
    plt.ylim(y_min, y_max)
    plt.xlabel('timestep')
    plt.ylabel('y_' + str(i * 5))
    plt.legend()
    plt.grid(True)
plt.show()

# for i in range(row * column):
#     axis = plt.subplot(row, column, i+1)
#     plt.plot(unc[i, :], 'r', label='uncertainty')
#     plt.ylim(unc[:8, :].min() - 0.5, unc[:8, :].max() + 0.5)
#     plt.xlabel('timestep')
#     plt.ylabel('unc_y_' + str(i * 5))
#     plt.legend()
#     plt.grid(True)
# plt.show()

# for i in range(row * column):
#     axis = plt.subplot(row, column, i+1)
#     plt.plot(unc[i+8, :], 'r', label='uncertainty')
#     plt.ylim(unc[8:, :].min() - 0.05, unc[8:, :].max() + 0.05)
#     plt.xlabel('timestep')
#     plt.ylabel('unc_eps_' + str(i * 5))
#     plt.legend()
#     plt.grid(True)
# plt.show()