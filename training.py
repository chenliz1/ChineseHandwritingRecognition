import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

import keras
from keras.models import Model, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from dataloader import trainGenerator, validGenerator
from model_9_layer_GSLRE import build_model_GSLRE


rootDir='./data/image_data'
trainDir = os.path.join(rootDir, 'train')
validDir = os.path.join(rootDir, 'test')

classSize = 500

batchSize = 128
val_batchSize = 128
width, height = 96, 96

vg = validGenerator(validDir, val_batchSize, 96, 96)
tg = trainGenerator(trainDir, batchSize, 96, 96)

model = build_model_GSLRE(classSize)

sgd = optimizers.SGD(lr=0.1, momentum=0.9, decay=3.3e-7, nesterov=True)
model.compile(loss='categorical_crossentropy',
        optimizer='SGD',
        metrics=['accuracy'])

checkpointer = keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5',
                                               verbose = 0, save_best_only=True, monitor='val_acc')
# sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
# K.set_value(sgd.lr, 0.5 * K.get_value(sgd.lr))

model.fit_generator(tg, epochs=1, verbose=1,
                    validation_data=vg,
                    callbacks=[checkpointer],
                    use_multiprocessing=True)