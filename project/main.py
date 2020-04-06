from custom_2D_unet import *
from custom_2D_unet_helpers import *
from helpers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

dataset_root_dir = '/workspace/datasets/kbc_sm/'
results_root_dir = '/workspace/results/kbc_sm/'

train_images, train_masks = load_and_preprocess_train_images_and_masks(dataset_root_dir + 'train', 'image', 'label')
evaluation_images, evaluation_masks = load_and_preprocess_evaluation_images_and_masks(dataset_root_dir + 'evaluation', 'image', 'label')

model = custom_unet((1024, 512, 1), num_classes=4, use_batch_norm=True, output_activation='softmax')
model.compile(optimizer='adam', loss=jaccard_distance, metrics=['accuracy'])
model.summary()

results_dir = create_results_dir_results_predict_dir_and_logs_dir(results_root_dir)
model_checkpoint = ModelCheckpoint(results_dir + 'unet_jaccard.hdf5', monitor='loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(results_dir + 'logs/', histogram_freq=1)
model.fit(train_images, train_masks, batch_size=3, epochs=50, callbacks=[model_checkpoint, tensorboard])

evaluation_loss, evaluation_accuracy = model.evaluate(evaluation_images, evaluation_masks)
print('Evaluation loss:' + str(evaluation_loss) + ' accuracy:' + str(evaluation_accuracy))
test_images = load_and_preprocess_test_images(dataset_root_dir + "test")
result_masks = model.predict(test_images, 1, verbose=1)
convert_results_to_gray_images_and_save(results_dir + "predict/", result_masks)
