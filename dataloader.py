from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle as pkl

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class HanziTrainingDataset(Dataset):

    def __init__(self, rootDir, charDictDir, transform=None):
        with open(charDictDir, 'rb') as handler:
            self.charDict = pkl.load(handler)
        self.rootDir = rootDir
        self.sampleNumList = np.zeros(len(self.charDict))

        for c in range(len(self.charDict)):
            samplesPath = os.path.join(self.rootDir, '{:05}'.format(c))
            self.sampleNumList[c] = np.int(len(os.listdir(samplesPath)))
        self.sampleCumList = np.int32(np.cumsum(self.sampleNumList))

    def __len__self(self):
        return self.sampleCumList[-1]

    def __getitem__(self):

        sample = {}

        return sample
trainDataset = HanziTrainingDataset(rootDir='./data/image_data/train', charDictDir='./data/char_dict')
print(len(trainDataset.charDict))
print(trainDataset.sampleCumList[-1])