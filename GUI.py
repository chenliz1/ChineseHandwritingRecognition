
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import cv2

width = 300
height = 300
center = height//2
white = (255, 255, 255)

# load model globally -> refer to script_Rio.ipynb

# for each output, new gui to display best 8 matches
# click on each one to select and move to text input area


# preprocess to binary and reshape
def preprocess(image, size=100):
    input_img = cv2.imread("image.png")
    # change to binary 0 - 1

    # reshape to 1 96 96 1


# ready to pass the image to the next function (probably our model)
def detect():
    filename = "image.png"
    image1.save(filename)
    preprocess(image1, 10)
    # pass to our model

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 0.5), (event.y - 0.5)
    x2, y2 = (event.x + 0.5), (event.y + 0.5)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
    draw.line([x1, y1, x2, y2],fill="black",width=10)

def delete():
    cv.delete("all")

root = Tk()

# create the canvas with specified width and height
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
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
