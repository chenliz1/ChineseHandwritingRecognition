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


def training_loop(model, train_data, valid_data, weights_path, history_path, epoch=50):
	'''
		epoch： must be a multiple of 10

	'''
	tg = train_data
	vg = valid_data
	checkpointer = keras.callbacks.ModelCheckpoint(filepath=weights_path,
											   verbose = 0, save_best_only=True, monitor='val_acc')

	record = {'acc': [], 'val_acc':[], 'loss':[], 'val_loss':[]}
	
	for i in range(epoch//10):
		tens = i + 1
		Lrate = 0.1**tens
		sgd = optimizers.SGD(lr=Lrate, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])
		history = model.fit_generator(tg, epochs=10, verbose=1,
							validation_data=vg,
							callbacks=[checkpointer])

		record['acc'] += history.history['acc']
		record['val_acc'] += history.history['val_acc']
		record['loss'] += history.history['loss']
		record['val_loss'] += history.history['val_loss']

	with open(history_path, 'wb') as handler:
		pkl.dump(record, handler, protocol=pkl.HIGHEST_PROTOCOL)



def evaluation(model, weights_path, valid_data):

	tg = train_data
	vg = valid_data

	model.load_weights(weights_path)
	model.compile(loss='categorical_crossentropy',
		optimizer='SGD',
		metrics=['accuracy'])

	test_loss, test_acc = test_model.evaluate_generator(vg, verbose=1)

	return test_loss, test_acc

