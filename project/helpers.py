import datetime
import os
import cv2
import matplotlib.pyplot as plt
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


def load_and_preprocess_test_images_and_masks(evaluation_path, image_folder, mask_folder, mask_pixel_values_aka_classes, shape=(512, 256)):
    print("\n# loading and preprocessing test images and masks")
    evaluation_image_paths = os.listdir(evaluation_path + '/' + image_folder)
    evaluation_mask_paths = os.listdir(evaluation_path + '/' + mask_folder)
    if len(evaluation_image_paths) != len(evaluation_mask_paths):
        raise Exception

    images = []
    masks = []
    for i, name in enumerate(evaluation_image_paths):
        update_progress(((i+1) / len(evaluation_image_paths)) * 100)
        image = cv2.imread(evaluation_path + '/' + image_folder + '/' + name, cv2.IMREAD_GRAYSCALE)
        image = resize_image(image, shape)
        image = image.reshape((shape[0], shape[1], 1))
        mask = cv2.imread(evaluation_path + '/' + mask_folder + '/' + name, cv2.IMREAD_GRAYSCALE)
        mask = resize_image(mask, shape)
        mask = convert_pixel_mask_to_multiclass_matirx_mask(mask, mask_pixel_values_aka_classes)
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)


def load_and_preprocess_train_images_and_masks(train_path, image_folder, mask_folder, mask_pixel_values_aka_classes, count=None, shape=(512, 256)):
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
        update_progress(((i+1) / len(train_image_paths)) * 100)
        image = cv2.imread(train_path + '/' + image_folder + '/' + name, cv2.IMREAD_GRAYSCALE)
        image = resize_image(image, shape)
        image = image.reshape((shape[0], shape[1], 1))
        mask = cv2.imread(train_path + '/' + mask_folder + '/' + name, cv2.IMREAD_GRAYSCALE)
        mask = resize_image(mask, shape)
        mask = convert_pixel_mask_to_multiclass_matirx_mask(mask, mask_pixel_values_aka_classes)
        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def convert_multiclass_matirx_masks_to_pixel_masks_and_save(predicted_path, result_masks, mask_pixel_values_aka_classes):
    print("\n# converting multiclass matirx masks to pixel masks and saving")
    for i, item in enumerate(result_masks):
        output = convert_multiclass_matirx_mask_to_pixel_mask(item, mask_pixel_values_aka_classes)
        cv2.imwrite(predicted_path + str(i) + '_mask.png', output)


def convert_one_class_images_to_pixel_images_and_save(save_path, images, shape=(512, 256)):
    print("\n# converting one class images to pixel images and saving")
    for i, image in enumerate(images):
        grayscale_image = image.reshape((shape[0], shape[1]))
        cv2.imwrite(save_path + str(i) + '_image.png', grayscale_image)


def convert_pixel_mask_to_multiclass_matirx_mask(pixel_mask, mask_pixel_values_aka_classes):
    multiclass_matirx_mask = np.zeros((pixel_mask.shape[0],
                                       pixel_mask.shape[1],
                                       len(mask_pixel_values_aka_classes)))
    for x, y in np.ndindex(pixel_mask.shape):
        value = pixel_mask[x][y]
        if value in mask_pixel_values_aka_classes:
            index = mask_pixel_values_aka_classes.index(value)
        else:
            index = np.abs(mask_pixel_values_aka_classes - value).argmin()
        multiclass_matirx_mask[x][y][index] = 1
    return multiclass_matirx_mask


def convert_multiclass_matirx_mask_to_pixel_mask(multiclass_matirx_mask, mask_pixel_values_aka_classes):
    pixel_mask = np.zeros((multiclass_matirx_mask.shape[0],
                           multiclass_matirx_mask.shape[1]))
    for x, y in np.ndindex(multiclass_matirx_mask.shape[:2]):
        index = np.argmax(multiclass_matirx_mask[x][y])
        pixel_mask[x][y] = mask_pixel_values_aka_classes[index]
    return pixel_mask


def resize_image(image, output_shape):
    return cv2.resize(image, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)


def make_file_and_write(file_path, text):
    file = open(file_path, "w")
    file.write(text)
    file.close()


def update_progress(progress_percentage):
    print('\r# {0}%'.format(round(progress_percentage, 2)), end='')


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
            :param titles:
            :param images:
            :param cols:
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
