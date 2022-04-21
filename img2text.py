import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import easyocr

#read img
img = cv2.imread("filename.png")
plt.imshow(img)

#convert orignal image file into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##convert grayscale with thresold
th, threshed = cv2.threshold(gray, 175, 255, cv2.THRESH_TRUNC)

#print(threshed)

#run reader with thai ('th') and english ('en')
reader = easyocr.Reader(['th','en'], gpu = False) # this needs to run only once to load the model into memory

#OCR with reader from easyocr module
text_ocr = reader.readtext(threshed)
print(text_ocr)

#define a function to draw bounding boxes
def draw_boxes(image, bounds, color = 'blue', width = 2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

#import image and draw bounding boxes on image
im = Image.open("filename")
draw_boxes(im, text_ocr)
