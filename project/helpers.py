import datetime
import os
import cv2
import numpy as np


def create_results_dir_results_predict_dir_and_logs_dir(root_dir):
    results_dir = root_dir + 'results/' + str(datetime.datetime.now()).replace(' ', '_')
    if not os.path.exists(results_dir):
        oldmask = os.umask(000)
        os.makedirs(results_dir, 0o777)
        os.umask(oldmask)
    results_dir_predict = results_dir + '/' + 'predictions'
    if not os.path.exists(results_dir_predict):
        oldmask = os.umask(000)
        os.makedirs(results_dir_predict, 0o777)
        os.umask(oldmask)
    results_dir_predict_image = results_dir + '/' + 'predictions/images'
    if not os.path.exists(results_dir_predict_image):
        oldmask = os.umask(000)
        os.makedirs(results_dir_predict_image, 0o777)
        os.umask(oldmask)
    results_dir_predict_mask = results_dir + '/' + 'predictions/masks'
    if not os.path.exists(results_dir_predict_mask):
        oldmask = os.umask(000)
        os.makedirs(results_dir_predict_mask, 0o777)
        os.umask(oldmask)
    results_dir_predict_results = results_dir + '/' + 'predictions/results'
    if not os.path.exists(results_dir_predict_results):
        oldmask = os.umask(000)
        os.makedirs(results_dir_predict_results, 0o777)
        os.umask(oldmask)
    results_dir_logs = results_dir + '/' + 'tensorboardlogs'
    if not os.path.exists(results_dir_logs):
        oldmask = os.umask(000)
        os.makedirs(results_dir_logs, 0o777)
        os.umask(oldmask)

    return results_dir + '/'


def load_and_preprocess_test_images_and_masks(evaluation_path, image_folder, mask_folder):
    print("\n# loading and preprocessing evaluation images and masks")
    evaluation_image_paths = os.listdir(evaluation_path + '/' + image_folder)
    evaluation_mask_paths = os.listdir(evaluation_path + '/' + mask_folder)
    if len(evaluation_image_paths) != len(evaluation_mask_paths):
        raise Exception

    images = []
    masks = []
    for i, name in enumerate(evaluation_image_paths):
        update_progress((i / len(evaluation_image_paths)) * 100)
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


def load_and_preprocess_train_images_and_masks(train_path, image_folder, mask_folder, count=None):
    print("\n# loading and preprocessing train images and masks")
    train_image_paths = os.listdir(train_path + '/' + image_folder)
    train_mask_paths = os.listdir(train_path + '/' + mask_folder)
    if count is not None:
        train_image_paths = train_image_paths[:count]
        train_mask_paths = train_mask_paths[:count]
    if len(train_image_paths) != len(train_mask_paths):
        raise Exception

    images = []
    masks = []
    for i, name in enumerate(train_image_paths):
        update_progress((i / len(train_image_paths)) * 100)
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
    print("\n# loading and preprocessing test images")
    images = []
    for test_image_name in os.listdir(test_path):
        image = cv2.imread(test_path + '/' + test_image_name, cv2.IMREAD_GRAYSCALE).reshape((1024, 512, 1))
        images.append(image)

    return np.array(images)


def convert_masks_to_gray_images_and_save(predicted_path, result_masks):
    print("\n# converting result masks to gray images and saving")
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
        cv2.imwrite(predicted_path + str(i) + '_mask.png', output)


def convert_results_to_gray_images(result_masks):
    print("\n# converting result masks to gray images")
    converted = []
    for i, item in enumerate(result_masks):
        update_progress((i / len(result_masks)) * 100)
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
        converted.append(output)
    return converted


def convert_images_to_gray_images_and_save(save_path, images):
    print("\n# converting images to gray images and saving")
    for i, image in enumerate(images):
        grayscale_image = image.reshape((1024, 512))
        cv2.imwrite(save_path + str(i) + '_image.png', grayscale_image)


def make_file_and_write(file_path, text):
    file = open(file_path, "w")
    file.write(text)
    file.close()


def update_progress(progress_percentage):
    print('\r# {0}%'.format(round(progress_percentage, 2)), end='')


import matplotlib.pyplot as plt
import numpy as np


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
