from helpers import *
import os
import cv2

# position working directory to generated dir from server

# model_path = 'unet_jaccard.hdf5'
# model = load_model(model_path, custom_objects={'jaccard_distance': jaccard_distance})
images = []
masks = []
results = []
for image, mask, result in zip(os.listdir('./predictions/images'), os.listdir('./predictions/masks'), os.listdir('./predictions/results')):
    images.append(cv2.imread('./predictions/images/' + image, cv2.IMREAD_GRAYSCALE))
    masks.append(cv2.imread('./predictions/masks/' + mask, cv2.IMREAD_GRAYSCALE))
    results.append(cv2.imread('./predictions/results/' + result, cv2.IMREAD_GRAYSCALE))
for original_image, original_mask, predicted_mask in zip(images, masks, results):
    show_images([original_image, original_mask, predicted_mask], titles=["original image", "original mask", "predicted mask"])
