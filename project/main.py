from custom_2D_unet import *
from custom_2D_unet_helpers import *
from helpers import *
from tensorflow.keras.callbacks import ModelCheckpoint

root_dir = '/workspace/datasets/kbc_sm/'

train_images, train_masks = load_and_preprocess_train_images(root_dir + 'train', 'image', 'label')

model = custom_unet((1024, 512, 1), num_classes=4, use_batch_norm=True, output_activation='softmax')
model.compile(optimizer='adam', loss=jaccard_distance, metrics=['accuracy'])
model.summary()

results_dir = create_results_dir_and_results_predict_dir('/workspace/results/kbc_sm/')
model_checkpoint = ModelCheckpoint(results_dir + 'unet_jaccard_3.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit(train_images, train_masks, batch_size=3, epochs=1000, callbacks=[model_checkpoint])

# TODO evaluation
test_images = load_and_preprocess_test_images(root_dir + "test")
result_masks = model.predict(test_images, 1, verbose=1)
convert_results_to_gray_images_and_save(results_dir + "predict/", result_masks)
