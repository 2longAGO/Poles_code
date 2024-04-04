import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pylab as plt
# get image
img = cv2.imread("C:/Users/leefl/Desktop/Poles_code/test.png") #C:/Users/leefl/Desktop/Poles_code/image_Y45.49420400065291X-75.65374787309156.jpg | C:/Users/leefl/Desktop/Poles_code/image_Y45.49448560622111X-75.65373866263494.jpg
# C:/Users/leefl/Desktop/Poles_code/test.png
print(img.shape[1])

# remove added black margins
if np.sum(img[:,-1]) == 0:
    print("bars")
    img[np.sum(a, axis=0)!=0]
    img[np.sum(a, axis=1)!=0]

resize = max(img.shape[0]//2016,1)
img = img[::resize,::resize]

# Load a model
model = YOLO('runs/segment/train26/weights/best.pt')  # load a pretrained model (recommended for training) 
results = model(img)  # predict on an image
for r in results:
    if r.masks != None :
        editArr = np.copy(r.masks.xy[0])
        #r.boxes.xywh[0][0]+r.boxes.xywh[0][2]/2
        box_y = (r.boxes.xywh[0][1]+r.boxes.xywh[0][3]//2).cpu().numpy()
        lineArr = editArr[ np.absolute(editArr[:,1]-box_y) < 30 ]
        print(lineArr)
        wdth = (lineArr[:,0].max() - lineArr[:,0].min())/img.shape[1]
        print(wdth)
        annotated = r.plot(boxes=False)
        plt.imshow(annotated)
        plt.axis('off')
plt.show()