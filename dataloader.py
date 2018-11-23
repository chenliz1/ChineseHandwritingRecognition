from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from skimage.transform import resize

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode
# plt.interactive(False)
#
# class HanziTrainingDataset(Sequence):
#
#     def __init__(self, rootDir, charDictDir, transform=None):
#         with open(charDictDir, 'rb') as handler:
#             self.charDict = pkl.load(handler)
#         self.rootDir = rootDir
#         self.sampleNumList = np.zeros(len(self.charDict))
#         self.sampleFilesDict = {}
#         self.transform = transform
#         for c in range(len(self.charDict)):
#             samplesPath = os.path.join(self.rootDir, '{:05}'.format(c))
#             self.sampleFilesDict[c] = []
#
#             for files in os.listdir(samplesPath):
#                 if files.endswith(".png"):
#                     self.sampleFilesDict[c].append(files)
#
#             self.sampleNumList[c] = np.int(len(self.sampleFilesDict[c]))
#         self.sampleCumList = np.int32(np.cumsum(self.sampleNumList))
#
#     def __len__self(self):
#         return self.sampleCumList[-1]
#
#     def __getitem__(self, idx):
#         label = (self.sampleCumList < idx).sum()
#         labelPath = '{:05}'.format(label)
#         if label > 0:
#             index = idx - self.sampleCumList[label-1]
#         else:
#             index = idx
#         fileIndex = self.sampleFilesDict[label][index]
#
#         imgName = os.path.join(self.rootDir, labelPath, fileIndex)
#         image = io.imread(imgName)
#         sample = {'image': image, 'label': label}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
# trainDataset = HanziTrainingDataset(rootDir='./data/image_data/train', charDictDir='./data/char_dict')
#
# print(trainDataset[8995]['label'])

# plt.imshow(trainDataset[8995]['image'].shape)
# plt.show()


# with open('./data/char_dict', 'rb') as handler:
#     charDict = pkl.load(handler)
# classSize = len(charDict)


def validGenerator(path, batchSize, width, height):
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    valid_generator = valid_datagen.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batchSize,
        color_mode="grayscale",
        class_mode='categorical')

    return valid_generator


def trainGenerator(path, batchSize, width, height):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05)

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batchSize,
        color_mode="grayscale",
        class_mode='categorical')

    return train_generator