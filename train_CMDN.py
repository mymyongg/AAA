from CMDN import CMDN
import os
import numpy as np
import cv2

if __name__ == "__main__":
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
        labels = labels.get('labels') # (N, L, 3)
        labels = np.concatenate([labels[:, 1:, 1], labels[:, 1:, 2]], axis=1)
        dataset_labels.append(labels)
    dataset_labels = np.concatenate(dataset_labels, axis=0) # (N, 10)

    num_dataset = dataset_images.shape[0]
    
    # Shuffle dataset
    shuffled_index = np.random.permutation(num_dataset)
    # dataset_images = dataset_images[shuffled_index][:20000, ...]
    # dataset_labels = dataset_labels[shuffled_index][:20000, ...]
    dataset_images = dataset_images[shuffled_index]
    dataset_labels = dataset_labels[shuffled_index]

    num_dataset = dataset_images.shape[0]
    print('num_dataset:{}'.format(num_dataset))

    num_train = int(0.8 * num_dataset)

    # Devide training and validation data
    x_train = dataset_images[:num_train, ...]
    y_train = dataset_labels[:num_train, ...]

    x_validation = dataset_images[num_train:, ...]
    y_validation = dataset_labels[num_train:, ...]

    # Preprocess input
    x_train = x_train / 255.0
    x_validation = x_validation / 255.0

    # Train model
    estimator = CMDN(img_size=(90, 320, 3), M=8, KMIX=3)
    estimator.build_model()
    estimator.train_model(x_train, y_train, x_validation, y_validation)
    estimator.report_training_results()
    estimator.save_training_results()