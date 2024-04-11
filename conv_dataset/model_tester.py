import cv2
from PIL import Image, ImageChops
import numpy as np
from ultralytics import YOLO
import matplotlib.pylab as plt
import sys

# https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil#10616717
def trim(im):
    bg = Image.new(im.mode, im.size, 0)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# get image
#img = cv2.imread("C:/Users/leefl/Desktop/Poles_code/image_Y45.50091361047674X-75.5628915706322.jpg") #C:/Users/leefl/Desktop/Poles_code/image_Y45.49420400065291X-75.65374787309156.jpg | C:/Users/leefl/Desktop/Poles_code/image_Y45.49448560622111X-75.65373866263494.jpg
# C:/Users/leefl/Desktop/Poles_code/test.png
img = Image.open("C:/Users/leefl/Desktop/Poles_code/conv_dataset/datasets/coco_converted/images/val/testPole5.png")
print(img.width)
img = trim(img)
img.thumbnail([sys.maxsize, 2048], Image.LANCZOS)

'''
# remove added black margins
if np.sum(img[:,-1]) == 0:
    print("bars")
    img[np.sum(a, axis=0)!=0]
    img[np.sum(a, axis=1)!=0]

resize = max(img.shape[0]//2016,1)
img = img[::resize,::resize]
'''
print(img.width)
# Load a model
model = YOLO('runs/segment/train42/weights/best.pt')  # load a pretrained model (recommended for training) 
results = model(img)  # predict on an image
for r in results:
    if r.masks != None :
        for i in range(len(r.masks.xy)):
            editArr = np.copy(r.masks.xy[i])
            print(i)
            #r.boxes.xywh[0][0]+r.boxes.xywh[0][2]/2
            box_y = (r.boxes.xywh[i][1]+r.boxes.xywh[i][3]//2).cpu().numpy()
            print(0.075*box_y)
            lineArr = editArr[ np.absolute(editArr[:,1]-box_y) < min((0.075*box_y),30) ]
            print(lineArr)
            wdth = (lineArr[:,0].max() - lineArr[:,0].min())/img.width
            print(wdth)
        annotated = r.plot(boxes=False) # boxes=False
        plt.imshow(annotated)
        plt.axis('off')

plt.show()