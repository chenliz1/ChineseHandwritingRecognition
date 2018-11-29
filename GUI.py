# TODO: delete the canvas content
# TODO: be able to resize the canvas
# TODO: gui on the given text (as button?, make some text input area)

# gui import
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import cv2

canvas_width = 96
canvas_height = 96
center = canvas_height//2
white = (255, 255, 255)

# model import

import numpy as np
np.set_printoptions(threshold=np.nan)
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

# load model globally -> refer to script_Rio.ipynb

# for each output, new gui to display best 8 matches
# click on each one to select and move to text input area


# preprocess to binary and reshape
def preprocess(image):
    # only need 1 channel, range from 0 to 1
    input_img = plt.imread("image.jpg")[:,:,0]//255.0
    # the array looks right
    count_0 = 0
    count_1 = 0
    for i in input_img:
        for j in i:
            if j == 0:
                count_0 += 1
            if j == 1:
                count_1 += 1
    print("1. num of 0/1:",count_0, count_1)
    input_img = np.resize(input_img, (1,96,96,1))

    count_0 = 0
    count_1 = 0
    for i in input_img[0]:
        for j in i:
            if j == 0:
                count_0 += 1
            if j == 1:
                count_1 += 1
    print("2. num of 0/1:", count_0, count_1)

    return input_img
# ready to pass the image to the next function (probably our model)
def detect():
    filename = "image.jpg"
    image1.save(filename)
    input_img = preprocess(image1)
    # pass to our model which has been loaded already
    print(input_img.mean())
    result = test_model.predict(input_img, batch_size=1).reshape(classSize)
    index = result.argsort()[-8:][::-1]
    for i in index:
        print(index_key_500[i])

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=3)
    draw.line([x1, y1, x2, y2],fill="black",width=3)

def delete():
    cv.delete("all")
    if os.path.isfile("image.jpg"):
        os.remove("image.jpg")

    image1 = PIL.Image.new("RGB", (canvas_width, canvas_height), white)

if __name__ == "__main__":

    # load train and test data, code from script.ipynb written by Lizhe Chen

    rootDir = './data/image_data'
    trainDir = os.path.join(rootDir, 'train')
    validDir = os.path.join(rootDir, 'test')
    with open('./data/char_dict', 'rb') as handler:
        charDict = pkl.load(handler)

    with open('./data/index_key_500.pickle', 'rb') as handler:
        index_key_500 = pkl.load(handler)

    classSize = 500
    batchSize = 128
    val_batchSize = 128
    width, height = 96, 96
    # vg = validGenerator(validDir, val_batchSize, 96, 96)
    vg = validGenerator(validDir, val_batchSize, 96, 96, binary=True, thresh=0.99)
    # # tg = trainGenerator(trainDir, batchSize, 96, 96)
    # tg = trainGenerator(validDir, val_batchSize, 96, 96, binary=True, thresh=0.99)

    test_model = build_model_GSLRE(500)
    test_model.load_weights('model_b0.99_weights_best_epoch.hdf5')

    # test_loss, test_acc = test_model.evaluate_generator(vg, verbose=1)
    # sample = vg[7][0][0].reshape(1, 96, 96, 1)
    #
    # result = test_model.predict(sample, batch_size=1).reshape(classSize)
    # # number of output
    # index = result.argsort()[-8:][::-1]
    # # dont need this one when in gui
    # # plt.imshow(sample.reshape(96, 96), cmap='gray')
    # #
    # for i in index:
    #     print(index_key_500[i])




    # logic to implement a python gui written by Yue Li

    root = Tk()

    # create the canvas with specified width and height
    cv = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    cv.pack()

    # PIL create an empty image and draw object to draw on
    # memory only, not visible
    image1 = PIL.Image.new("RGB", (canvas_width, canvas_height), white)
    draw = ImageDraw.Draw(image1)

    # do the Tkinter canvas drawings (visible)
    # cv.create_line([0, center, width, center], fill='green')

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    # do the PIL image/draw (in memory) drawings
    # draw.line([0, center, width, center], green)

    # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
    # filename = "my_drawing.png"
    # image1.save(filename)

    button_save=Button(text="detect",command=detect)
    button_delete=Button(text="clear",command=delete)

    button_save.pack()
    button_delete.pack()
    root.mainloop()
