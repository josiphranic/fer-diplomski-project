from custom_2D_unet import *
from custom_2D_unet_helpers import *
from helpers import *
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint


root_dir = '/workspace/datasets/kbc_sm/'
results_dir = create_results_dir_and_results_predict_dir('/workspace/results/kbc_sm/')

BATCH_SIZE = 4
TRAIN_SIZE = 1133
SPE = ceil(TRAIN_SIZE / BATCH_SIZE)
myGene = trainGenerator(root_dir + 'train','image','label',batch_size=BATCH_SIZE)

model = custom_unet((1024, 512, 1), num_classes=4, output_activation='softmax')
model.compile(optimizer = 'adam', loss = jaccard_distance, metrics = ['accuracy'])

model_checkpoint = ModelCheckpoint(results_dir + 'unet_jaccard_3.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=SPE,epochs=30,callbacks=[model_checkpoint])

testGene = testGenerator(root_dir + "test")
results = model.predict_generator(testGene,1,verbose=1)
saveResult(results_dir + "predict/", results)