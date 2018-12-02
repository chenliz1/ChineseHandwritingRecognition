import keras
from keras.models import Model, load_model
from keras import backend as K
from model_9_layer_GSLRE import build_model_GSLRE
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

def binarify_sample(rescaled_image, threshold):
	'''
		rescaled_image: one channel image with rescaled value [0, 1]
	'''
	result = rescaled_image.reshape(96, 96)
	result[result<=0.99] = 0
	result[result>0.99] = 1

	return result

def layer_predict(model, layername, data):
	'''
		For conv layers only.
		data: with shape batch_size * 96 * 96 * 1
		layername: start with (conv)
	'''
	intermediate_layer_model = Model(inputs=model.input,
								 outputs=model.get_layer(layername).output)

	intermediate_output = intermediate_layer_model.predict(data)
	return intermediate_output


def display_channel_images(model, layername, data, column_number=10):
	'''
		take the ouyput from layer_predict
		column_number: the width of the output will be column_number x width of layer_output
	'''

	layer_output = layer_predict(model, layername, data)
	batchSize, height, width, channels = layer_output.shape
	canvas = np.zeros((batchSize, height * ((channels//column_number) + 1), width * column_number))
	for i in range(batchSize):
		for j in range(channels):
			image = layer_output[i, :, :, j].reshape(height, width)
			H_S = (j // column_number) * height
			W_S = (j % column_number) * width
			canvas[i, H_S: (H_S+height), W_S: (W_S+width)] = image

	return canvas

def compare_curves(modelA_history_path, modelB_history_path, save_path):
	'''
		compare the curves of Model Origin and Model B0.99
		suppose both models traind with same number of epoches
	'''
	with open(modelA_history_path, 'rb') as file:
		recordA = pkl.load(file)

	with open(modelB_history_path, 'rb') as file:
		recordB = pkl.load(file)

	epoch = max(len(recordA['acc']), len(recordB['acc']))

	acc_record = {'acc': [], 'val_acc':[],'acc-binary': [], 'val_acc-binary':[] }
	loss_record = {'loss':[], 'val_loss':[], 'loss-binary': [], 'val_loss-binary':[]}

	acc_record['acc'] = recordA['acc']
	acc_record['val_acc'] = recordA['val_acc']
	acc_record['acc-binary'] = recordB['acc']
	acc_record['val_acc-binary'] = recordB['val_acc']

	loss_record['loss'] = recordA['loss']
	loss_record['val_loss'] = recordA['val_loss']
	loss_record['loss-binary'] = recordB['loss']
	loss_record['val_loss-binary'] = recordB['val_loss']

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes[0].plot([x for x in range(0, epoch)], acc_record['acc'])
	axes[0].plot([x for x in range(0, epoch)], acc_record['val_acc'])
	axes[0].plot([x for x in range(0, epoch)], acc_record['acc-binary'])
	axes[0].plot([x for x in range(0, epoch)], acc_record['val_acc-binary'])

	axes[1].plot([x for x in range(0, epoch)], loss_record['loss'])
	axes[1].plot([x for x in range(0, epoch)], loss_record['val_loss'])
	axes[1].plot([x for x in range(0, epoch)], loss_record['loss-binary'])
	axes[1].plot([x for x in range(0, epoch)], loss_record['val_loss-binary'])


	axes[0].set_title('Model Accuracy')
	axes[0].set_ylabel('Accuracy')
	axes[0].set_xlabel('Epoch')
	axes[0].legend(['Train(Original Input)', 'Validation(Original Input)', 
					'Train(Binary Input)', 'Validation(Binary Input)'], loc='lower right')

	axes[1].set_title('Model Loss')
	axes[1].set_ylabel('Loss')
	axes[1].set_xlabel('Epoch')
	axes[1].legend(['Train(Original Input)', 'Validation(Original Input)', 
					'Train(Binary Input)', 'Validation(Binary Input)'], loc='upper right')

	fig.savefig(save_path, dpi=200 )


def plotting_curves(history_path, save_path):
	'''
		ploting training/validation loss/accuracy curves
	'''

	with open(history_path, 'rb') as file:
		record = pkl.load(file)

	epoch = len(record['acc'])
	fig, axes = plt.subplots(1, 2, figsize=(15, 7)) 
	axes[0].plot([x for x in range(0, epoch)], record['acc'])
	axes[0].plot([x for x in range(0, epoch)], record['val_acc'])
	axes[1].plot([x for x in range(0, epoch)], record['loss'])
	axes[1].plot([x for x in range(0, epoch)], record['val_loss'])


	axes[0].set_title('Model Accuracy')
	axes[0].set_ylabel('Accuracy')
	axes[0].set_xlabel('Epoch')
	axes[0].legend(['Train', 'Validation'], loc='lower right')

	axes[1].set_title('Model Loss')
	axes[1].set_ylabel('Loss')
	axes[1].set_xlabel('Epoch')
	axes[1].legend(['Train', 'Validation'], loc='upper right')

	fig.savefig(save_path, dpi='figure' )




