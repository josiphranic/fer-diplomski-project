from tensorflow.keras.models import load_model
from helpers import *
from custom_2D_unet_helpers import *

# position working directory to generated dir from server

model_path = 'unet_jaccard.hdf5'

model = load_model(model_path, custom_objects={'jaccard_distance': jaccard_distance})
evaluation_images, evaluation_masks = load_and_preprocess_evaluation_images_and_masks('evaluation', 'image', 'label')
model.evaluate(evaluation_images, evaluation_masks)
predicted_masks = model.predict(evaluation_images, 1, verbose=1)
evaluation_masks_converted_to_gray = convert_results_to_gray_images(evaluation_masks)
predicted_and_converted_masks = convert_results_to_gray_images(predicted_masks)
evaluation_images_grayscale = []
for e in evaluation_images:
    evaluation_images_grayscale.append(e.reshape((1024, 512)))

for original_image, original_mask, predicted_mask in zip(evaluation_images_grayscale, evaluation_masks_converted_to_gray, predicted_and_converted_masks):
    show_images([original_image, original_mask, predicted_mask], titles=["original image", "original mask", "predicted mask"])
