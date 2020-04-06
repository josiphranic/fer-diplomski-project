import datetime
import os
import cv2
import numpy as np


def create_results_dir_and_results_predict_dir(root_dir):
    results_dir = root_dir + 'results/' + str(datetime.datetime.now()).replace(' ', '_')
    if not os.path.exists(results_dir):
        oldmask = os.umask(000)
        os.makedirs(results_dir, 0o777)
        os.umask(oldmask)
    results_dir_predict = results_dir + '/' + 'predict'
    if not os.path.exists(results_dir_predict):
        oldmask = os.umask(000)
        os.makedirs(results_dir_predict, 0o777)
        os.umask(oldmask)

    return results_dir + '/'


def load_and_preprocess_evaluation_images_and_masks(evaluation_path, image_folder, mask_folder):
    print("# loading and preprocessing evaluation images and masks")
    evaluation_image_paths = os.listdir(evaluation_path + '/' + image_folder)
    evaluation_mask_paths = os.listdir(evaluation_path + '/' + mask_folder)
    if len(evaluation_image_paths) != len(evaluation_mask_paths):
        raise Exception

    images = []
    masks = []
    for name in evaluation_image_paths:
        image = cv2.imread(evaluation_path + '/' + image_folder + '/' + name, cv2.IMREAD_GRAYSCALE).reshape((1024, 512, 1))
        mask = cv2.imread(evaluation_path + '/' + mask_folder + '/' + name, cv2.IMREAD_GRAYSCALE)

        # TODO resize
        output_mask = np.zeros((1024, 512, 4))
        for x, y in np.ndindex(mask.shape):
            value = mask[x][y]
            if value == 0:
                output_mask[x][y][0] = 1
            elif value == 45:
                output_mask[x][y][1] = 1
            elif value == 125:
                output_mask[x][y][2] = 1
            elif value == 205:
                output_mask[x][y][3] = 1
            else:
                raise Exception
        images.append(image)
        masks.append(output_mask)
    return np.array(images), np.array(masks)


def load_and_preprocess_train_images(train_path, image_folder, mask_folder):
    print("# loading and preprocessing images")
    train_image_paths = os.listdir(train_path + '/' + image_folder)
    train_mask_paths = os.listdir(train_path + '/' + mask_folder)
    if len(train_image_paths) != len(train_mask_paths):
        raise Exception

    images = []
    masks = []
    for name in train_image_paths:
        image = cv2.imread(train_path + '/' + image_folder + '/' + name, cv2.IMREAD_GRAYSCALE).reshape((1024, 512, 1))
        mask = cv2.imread(train_path + '/' + mask_folder + '/' + name, cv2.IMREAD_GRAYSCALE)

        # TODO resize
        output_mask = np.zeros((1024, 512, 4))
        for x, y in np.ndindex(mask.shape):
            value = mask[x][y]
            if value == 0:
                output_mask[x][y][0] = 1
            elif value == 45:
                output_mask[x][y][1] = 1
            elif value == 125:
                output_mask[x][y][2] = 1
            elif value == 205:
                output_mask[x][y][3] = 1
            else:
                raise Exception
        images.append(image)
        masks.append(output_mask)

    return np.array(images), np.array(masks)


def load_and_preprocess_test_images(test_path):
    print("# loading and preprocessing test images")
    images = []
    for test_image_name in os.listdir(test_path):
        image = cv2.imread(test_path + '/' + test_image_name, cv2.IMREAD_GRAYSCALE).reshape((1024, 512, 1))
        images.append(image)

    return np.array(images)


def convert_results_to_gray_images_and_save(predicted_path, result_masks):
    print("# converting result masks to gray images and saving")
    for i, item in enumerate(result_masks):
        output = np.zeros((1024, 512))
        for x in range(0, item.shape[0]):
            for y in range(0, item.shape[1]):
                value = np.argmax(item[x][y])
                if value == 0:
                    output[x][y] = 0
                elif value == 1:
                    output[x][y] = 45
                elif value == 2:
                    output[x][y] = 125
                elif value == 3:
                    output[x][y] = 205
                else:
                    raise Exception
        cv2.imwrite(predicted_path + str(i) + '_predict_3.png', output)
