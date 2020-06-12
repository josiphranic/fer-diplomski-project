from comet_ml import Experiment
from custom_2D_unet import *
from custom_2D_unet_helpers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from image_augmentation_helper import *


# helpers
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


def load_and_preprocess_test_images_and_masks(evaluation_path, image_folder, mask_folder, mask_pixel_values_aka_classes, count=None, shape=(512, 256)):
    print("\n# loading and preprocessing test images and masks")
    evaluation_image_paths = os.listdir(evaluation_path + '/' + image_folder)
    evaluation_mask_paths = os.listdir(evaluation_path + '/' + mask_folder)
    evaluation_image_paths.sort()
    evaluation_mask_paths.sort()
    if count is not None:
        evaluation_image_paths = evaluation_image_paths[:count]
        evaluation_mask_paths = evaluation_mask_paths[:count]
    if evaluation_image_paths != evaluation_mask_paths:
        raise Exception("Test data invalid")

    image_and_mask_preprocess_data = [(name, evaluation_path, image_folder, mask_folder, shape, mask_pixel_values_aka_classes, False) for name in evaluation_image_paths]
    pool = Pool(processes=10)
    images_and_masks = pool.map(load_and_preprocess_image_and_mask, image_and_mask_preprocess_data)

    return np.array([y for x in [image_and_mask[0] for image_and_mask in images_and_masks] for y in x]), \
           np.array([y for x in [image_and_mask[1] for image_and_mask in images_and_masks] for y in x])


def load_and_preprocess_train_images_and_masks(train_path, image_folder, mask_folder, mask_pixel_values_aka_classes, count=None, shape=(512, 256)):
    print("\n# loading and preprocessing train images and masks")
    train_image_paths = os.listdir(train_path + '/' + image_folder)
    train_mask_paths = os.listdir(train_path + '/' + mask_folder)
    train_image_paths.sort()
    train_mask_paths.sort()
    if count is not None:
        train_image_paths = train_image_paths[:count]
        train_mask_paths = train_mask_paths[:count]
    if train_image_paths != train_mask_paths:
        raise Exception("Train data invalid")

    image_and_mask_preprocess_data = [(name, train_path, image_folder, mask_folder, shape, mask_pixel_values_aka_classes, True) for name in train_image_paths]
    pool = Pool(processes=10)
    images_masks_augmentation_and_name = pool.map(load_and_preprocess_image_and_mask, image_and_mask_preprocess_data)
    # log_augmented_images(images_masks_augmentation_and_name)

    return np.array([y for x in [image_and_mask[0] for image_and_mask in images_masks_augmentation_and_name] for y in x]), \
           np.array([y for x in [image_and_mask[1] for image_and_mask in images_masks_augmentation_and_name] for y in x])


def load_and_preprocess_image_and_mask(image_and_mask_preprocess_data):
    name, folder_path, image_folder, mask_folder, shape, mask_pixel_values_aka_classes, augment_data = image_and_mask_preprocess_data
    image = cv2.imread(folder_path + '/' + image_folder + '/' + name, cv2.IMREAD_GRAYSCALE)
    if shape[2] != 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.
    image = resize_image(image, shape)
    image = image.reshape((shape[0], shape[1], shape[2]))
    np_image = np.copy(image)
    augmented_images = get_augmented_images(image)
    if augment_data:
        image = augmented_images
    else:
        image = [image]
    mask = cv2.imread(folder_path + '/' + mask_folder + '/' + name, cv2.IMREAD_GRAYSCALE)
    mask = resize_image(mask, shape)
    mask = mask.reshape((shape[0], shape[1], 1))
    np_mask = np.copy(mask)
    augmented_masks = get_augmented_images(mask)
    mask = convert_pixel_mask_to_multiclass_matirx_mask(mask, mask_pixel_values_aka_classes)
    if augment_data:
        mask = [convert_pixel_mask_to_multiclass_matirx_mask(mask, mask_pixel_values_aka_classes) for mask in augmented_masks]
    else:
        mask = [mask]
    augmentation = (augmented_images, augmented_masks)
    return image, mask, augmentation, name, np_mask, np_image


def log_augmented_images(images_masks_augmentation_and_name):
    import time
    for image, mask, augmentation, name, np_mask, np_image in images_masks_augmentation_and_name:
        time.sleep(0.08)
        augmentation_images, augmentation_masks = augmentation
        experiment.log_image(np_image, name='augmented_' + name.replace('.png', '') + '_original_image')
        experiment.log_image(np_mask, name='augmented_' + name.replace('.png', '') + '_original_mask')
        [experiment.log_image(augmented_image[1], name='augmented_' + name.replace('.png', '') + '_' + str(augmented_image[0]) + '_image') for augmented_image in enumerate(augmentation_images)]
        [experiment.log_image(augmented_mask[1], name='augmented_' + name.replace('.png', '') + '_' + str(augmented_mask[0]) + '_mask') for augmented_mask in enumerate(augmentation_masks)]


def convert_multiclass_matirx_masks_to_pixel_masks_and_save(predicted_path, result_masks, mask_pixel_values_aka_classes):
    print("\n# converting multiclass matirx masks to pixel masks and saving")
    results = []
    for i, item in enumerate(result_masks):
        output = convert_multiclass_matirx_mask_to_pixel_mask(item, mask_pixel_values_aka_classes)
        cv2.imwrite(predicted_path + str(i) + '_mask.png', output)
        results.append(output)
    return results


def convert_one_class_images_to_pixel_images_and_save(save_path, images, shape=(512, 256)):
    print("\n# converting one class images to pixel images and saving")
    results = []
    for i, image in enumerate(images):
        image = np.rint(image * 255)
        grayscale_image = image.reshape((shape[0], shape[1], shape[2]))
        cv2.imwrite(save_path + str(i) + '_image.png', grayscale_image)
        results.append(grayscale_image)
    return results


def convert_pixel_mask_to_multiclass_matirx_mask(pixel_mask, mask_pixel_values_aka_classes):
    multiclass_matirx_mask = np.zeros((pixel_mask.shape[0],
                                       pixel_mask.shape[1],
                                       len(mask_pixel_values_aka_classes)),
                                      dtype=np.uint8)
    for x, y in np.ndindex(pixel_mask.shape[:2]):
        # some images after augmentation are in 2d shape and some in 3d
        value = max(pixel_mask[x][y], 0)
        if value in mask_pixel_values_aka_classes:
            index = mask_pixel_values_aka_classes.index(value)
        else:
            index = np.abs(mask_pixel_values_aka_classes - value).argmin()
        multiclass_matirx_mask[x][y][index] = 1
    return multiclass_matirx_mask


def convert_multiclass_matirx_mask_to_pixel_mask(multiclass_matirx_mask, mask_pixel_values_aka_classes):
    pixel_mask = np.zeros((multiclass_matirx_mask.shape[0],
                           multiclass_matirx_mask.shape[1]),
                          dtype=np.uint8)
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


# main
dataset_root_dir = '/workspace/datasets/kbc_sm_splited/'
results_root_dir = '/workspace/results/kbc_sm/'
name = 'random weights, kbc_sm - unet'
# pretrained_model_path = '/workspace/results/sfu/results/2020-05-18_02:44:31.486451/unet.hdf5'
pretrained_model_path = '**VGG19'
trainable_encoder = True
random_weights = True
input_shape = (512, 256, 3)
mask_pixel_values_aka_classes = [0, 45, 125, 205]
# mask_pixel_values_aka_classes = [0, 64, 80, 100, 120, 192, 255]
number_of_classes = len(mask_pixel_values_aka_classes)
batch_size = 3
epochs = 1000
early_stopping_patience = 3
loss_function = jaccard_loss
validation_split = 0.15
output_activation = 'softmax'
description = 'epochs:' + str(epochs) + ' input shape:' + str(input_shape) + ' number of classes:' + str(number_of_classes) + ' batch size:' + str(batch_size)
description += ' early stopping patience:' + str(early_stopping_patience)  # + ' pretrained model:' + pretrained_model_path
description += ' loss function:' + loss_function.__name__
experiment = Experiment(api_key="51ASN1AUkvtHKtCGOFQsoEMWS", project_name="fer-diplomski-project", workspace="josiphranic")
experiment.set_name(name)
experiment.log_parameter("input_shape", str(input_shape))
experiment.log_parameter("mask_pixel_values_aka_classes", mask_pixel_values_aka_classes)
experiment.log_parameter("number_of_classes", number_of_classes)
experiment.log_parameter("batch_size", batch_size)
experiment.log_parameter("epochs", epochs)
experiment.log_parameter("early_stopping_patience", early_stopping_patience)
experiment.log_parameter("loss_function", loss_function.__name__)
experiment.log_parameter("validation_split", validation_split)
experiment.log_parameter("output_activation", output_activation)
experiment.log_parameter("trainable_encoder", trainable_encoder)
experiment.log_parameter("pretrained_model_path", pretrained_model_path)
experiment.log_parameter("random_weights", random_weights)

train_images, train_masks = load_and_preprocess_train_images_and_masks(dataset_root_dir + 'train', 'image', 'label', mask_pixel_values_aka_classes, shape=input_shape)
test_images, test_masks = load_and_preprocess_test_images_and_masks(dataset_root_dir + 'test', 'image', 'label', mask_pixel_values_aka_classes, shape=input_shape)

# model = custom_unet(input_shape, num_classes=number_of_classes, use_batch_norm=True, output_activation=output_activation)
# model = get_custom_model_with_pretrained_encoder(pretrained_model_path, number_of_classes, trainable_encoder=trainable_encoder)
model = custom_unet_with_vgg19_encoder(input_shape, num_classes=number_of_classes, use_batch_norm=True, output_activation=output_activation, trainable_encoder=trainable_encoder, random_weights=random_weights)
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy', dice_coef])
model.summary()

results_dir = create_results_dir_results_predict_dir_and_logs_dir(results_root_dir)
experiment.log_parameter("results_directory", results_dir)
model_checkpoint = ModelCheckpoint(results_dir + 'unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(results_dir + 'tensorboardlogs/', histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=early_stopping_patience)
model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint, tensorboard, early_stopping], validation_split=validation_split, shuffle=True)

print('Evaluation:')
evaluation_loss, evaluation_accuracy, evaluation_dice_coef = model.evaluate(test_images, test_masks)
experiment.log_metric("evaluation_loss", evaluation_loss)
experiment.log_metric("evaluation_accuracy", evaluation_accuracy)
experiment.log_metric("evaluation_dice_coef", evaluation_dice_coef)
make_file_and_write(results_dir + 'predictions/predictions_result.txt', 'evaluation loss:' + str(evaluation_loss)
                    + ' evaluation accuracy:' + str(evaluation_accuracy) + ' evaluation dice coef:' + str(evaluation_dice_coef))
make_file_and_write(results_dir + 'description.txt', description)

predicted_masks = model.predict(test_images, 1, verbose=1)
converted_test_images = convert_one_class_images_to_pixel_images_and_save(results_dir + 'predictions/images/', test_images, shape=input_shape)
converted_test_masks = convert_multiclass_matirx_masks_to_pixel_masks_and_save(results_dir + 'predictions/masks/', test_masks, mask_pixel_values_aka_classes)
converted_predicted_masks = convert_multiclass_matirx_masks_to_pixel_masks_and_save(results_dir + 'predictions/results/', predicted_masks, mask_pixel_values_aka_classes)

plot_model(model, to_file=results_dir + 'model_architecture.png', show_shapes=True, show_layer_names=True, rankdir='TB')
experiment.log_image(results_dir + 'model_architecture.png', name='model_architecture.png')
experiment.log_asset(results_dir + 'unet.hdf5', file_name='unet.hdf5')
for index in range(len(test_images)):
    experiment.log_image(converted_test_images[index], name=str(index) + '_test_image')
    experiment.log_image(converted_test_masks[index], name=str(index) + '_test_mask')
    experiment.log_image(converted_predicted_masks[index], name=str(index) + '_predicted_mask')
