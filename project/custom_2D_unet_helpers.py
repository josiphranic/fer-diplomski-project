import os
import cv2
import numpy as np
from tensorflow.keras import backend as K


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def trainGenerator(train_path, image_folder, mask_folder, batch_size=1):
    train_images = os.listdir(train_path + '/' + image_folder)
    train_masks = os.listdir(train_path + '/' + mask_folder)
    if len(train_images) != len(train_masks):
        raise Exception

    while True:
        batch_x = np.empty((batch_size, 1024, 512, 1))
        batch_y = np.empty((batch_size, 1024, 512, 4))
        batch_counter = 0
        batch_count = 0
        for name in train_images:
            image = cv2.imread(train_path + '/' + image_folder + '/' + name, cv2.IMREAD_UNCHANGED) / 205
            mask = cv2.imread(train_path + '/' + mask_folder + '/' + name, cv2.IMREAD_UNCHANGED)
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

            x, y = np.expand_dims(image, axis=0).reshape((1, 1024, 512, 1)), np.expand_dims(output_mask,
                                                                                            axis=0).reshape(
                (1, 1024, 512, 4))
            batch_x[batch_counter] = x
            batch_y[batch_counter] = y
            batch_counter += 1

            if batch_counter == batch_size:
                ret_x, ret_y = batch_x, batch_y
                batch_x = np.empty((batch_size, 1024, 512, 1))
                batch_y = np.empty((batch_size, 1024, 512, 4))
                batch_counter = 0
                batch_count += 1
                # print("Batch {}".format(batch_count))
                yield ret_x, ret_y

            if batch_count != 0:
                yield batch_x, batch_y


def testGenerator(test_path):
    for test_image in os.listdir(test_path):
        image = cv2.imread(test_path + '/' + test_image, cv2.IMREAD_UNCHANGED) / 205

        yield np.expand_dims(image, axis=0).reshape((1, 1024, 512, 1))


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        output = np.zeros((1024, 512))
        for x in range(0, item.shape[0]):
            for y in range(0, item.shape[1]):
                value = np.argmax(item[x][y])
                # value = np.delete(item[x][y], value)
                # value = np.argmax(value)
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
        #np.savetxt(save_path + str(i) + '_predict_3.csv', output, delimiter=",")
        cv2.imwrite(save_path + str(i) + '_predict_3.png', output)
