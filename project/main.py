from custom_2D_unet import *
from custom_2D_unet_helpers import *
from helpers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model

dataset_root_dir = '/workspace/datasets/kbc_sm/'
results_root_dir = '/workspace/results/kbc_sm/'

train_images, train_masks = load_and_preprocess_train_images_and_masks(dataset_root_dir + 'train', 'image', 'label')
test_images, test_masks = load_and_preprocess_test_images_and_masks(dataset_root_dir + 'test', 'image', 'label')

model = custom_unet((1024, 512, 1), num_classes=4, use_batch_norm=True, output_activation='softmax')
model.compile(optimizer='adam', loss=jaccard_distance, metrics=['accuracy'])
model.summary()

results_dir = create_results_dir_results_predict_dir_and_logs_dir(results_root_dir)
model_checkpoint = ModelCheckpoint(results_dir + 'unet_jaccard.hdf5', monitor='loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(results_dir + 'tensorboardlogs/', histogram_freq=1)
model.fit(train_images, train_masks, batch_size=4, epochs=30, callbacks=[model_checkpoint, tensorboard], validation_split=0.1, shuffle=True)

print('Evaluation:')
evaluation_loss, evaluation_accuracy = model.evaluate(test_images, test_masks)
make_file_and_write(results_dir + 'predictions/predictions_result.txt',
                    'evaluation loss:' + str(evaluation_loss) + ' evaluation accuracy:' + str(evaluation_accuracy))

predicted_masks = model.predict(test_images, 1, verbose=1)
convert_images_to_gray_images_and_save(results_dir + 'predictions/images/', test_images)
convert_masks_to_gray_images_and_save(results_dir + 'predictions/masks/', test_masks)
convert_masks_to_gray_images_and_save(results_dir + 'predictions/results/', predicted_masks)

plot_model(model, to_file=results_dir + 'model_architecture.png', show_shapes=True, show_layer_names=True, rankdir='LR')
