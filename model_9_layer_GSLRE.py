
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset
# from torchvision import transforms
# from torchvision.utils import save_image
# from torchvision.datasets import MNIST
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.misc
import matplotlib

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU
from keras.models import Model, load_model
from keras import backend as K
import pickle as pkl


# class HZCNN(nn.Module):
#     def __init__(self, kernel, num_filters, num_colours):
#         super(HZCNN, self).__init__()
#         padding = 3 // 2
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.BatchNorm2d(num_filters),
#             nn.PReLU())





def channelD(C, N, W, W_p, K=3, x=4):

    return (C * N * K * W_p) // ((C * W + N * W_p) * x)


def build_model_GSLRE(classSize, x=4):
    model = keras.Sequential()
    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', name='conv1',
                                   input_shape=(96, 96, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))


    D = channelD(96, 128, 48, 48, 3, x)
    model.add(Conv2D(filters=D, kernel_size=(3, 1), padding='same', name='conv2_de1'))
    model.add(Conv2D(filters=128, kernel_size=(1, 3), padding='same', name='conv2_de2'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    D = channelD(128, 160, 24, 24, 3, x)
    model.add(Conv2D(filters=D, kernel_size=(3, 1), padding='same', name='conv3_de1'))
    model.add(Conv2D(filters=160, kernel_size=(1, 3), padding='same', name='conv3_de2'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    D = channelD(160, 256, 12, 12, 3, x)
    model.add(Conv2D(filters=D, kernel_size=(3, 1), padding='same', name='conv4_1_de1'))
    model.add(Conv2D(filters=256, kernel_size=(1, 3), padding='same', name='conv4_1_de2'))
    model.add(BatchNormalization())
    model.add(PReLU())

    D = channelD(256, 256, 12, 12, 3, x)
    model.add(Conv2D(filters=D, kernel_size=(3, 1), padding='same', name='conv4_2_de1'))
    model.add(Conv2D(filters=256, kernel_size=(1, 3), padding='same', name='conv4_2_de2'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    D = channelD(256, 384, 6, 6, 3, x)
    model.add(Conv2D(filters=D, kernel_size=(3, 1), padding='same', name='conv5_1_de1'))
    model.add(Conv2D(filters=384, kernel_size=(1, 3), padding='same', name='conv5_1_de2'))
    model.add(BatchNormalization())
    model.add(PReLU())

    D = channelD(384, 384, 6, 6, 3, x)
    model.add(Conv2D(filters=D, kernel_size=(3, 1), padding='same', name='conv5_2_de1'))
    model.add(Conv2D(filters=384, kernel_size=(1, 3), padding='same', name='conv5_2_de2'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, name='fc1'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(classSize, activation='softmax', name='fc2'))

    return model


def build_model(classSize, x=4):
    model = keras.Sequential()
    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', name='conv1',
                                   input_shape=(96, 96, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))


    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv2'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))


    model.add(Conv2D(filters=160, kernel_size=(3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))


    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv4_1'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv4_2'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))


    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', name='conv5_1'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', name='conv5_2'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, name='fc1'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(classSize, activation='softmax', name='fc2'))

    return model

