# 446484.85,5035684.12 
# 38m//
'''
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6111250/
Pole detection dataset: https://universe.roboflow.com/mmutm/utility-pole-oxpzi/dataset/11
Custom dataset annotate with: https://www.makesense.ai/
tutorial: https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/
1st attempt: make a grid of all the points that encompass gatineau
2nd attempt: make a point cloud that is bound by the roads of gatineau
Google maps CRS = EPSG:4326 - WGS 84
'''

from streetview import search_panoramas, get_panorama # https://pypi.org/project/streetview/
from math import asin, atan2, cos, degrees, radians, sin
from ultralytics import YOLO
from os.path import exists
from PIL import Image, ImageChops
import matplotlib.pylab as plt
import numpy as np
import csv
import cv2
import sys
import re

model = YOLO('conv_dataset/runs/segment/train42/weights/best.pt')  # load a pretrained model (use best.pt)
# conv_dataset/runs/segment/train26/weights/best.pt

# Source: https://stackoverflow.com/a/7835325
def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2),)

# https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil#10616717
def trim(im):
    bg = Image.new(im.mode, im.size, 0)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

lat = 52.20472
lon = 0.14056
distance = 15
bearing = 90
lat2, lon2 = get_point_at_distance(lat, lon, distance, bearing)

with open('PanosData_.txt') as in_file:
    line_count = 0
    with open('PolePoints_1000.csv', 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',')
        csv_writer.writerow(['Y','X','pole_width','distance','origin_y','origin_x'])
        for row in in_file.readlines():
            print(line_count)
            #print(f'\t coords: Y: {row[0]} X: {row[1]}')
            row_arr = row.split(' ')
            if len(row_arr) == 0 :
                break
            latest = lambda: None
            latest.pano_id = row_arr[0].split("=")[1]
            latest.lat = float(row_arr[1].split("=")[1])
            latest.lon = float(row_arr[2].split("=")[1])
            latest.heading = float(row_arr[3].split("=")[1])
            latest.pitch = float(row_arr[4].split("=")[1])
            latest.roll = float(row_arr[5].split("=")[1])
            latest.date = None
            image = None
            if exists(f'image_Y{latest.lat}X{latest.lon}.jpg') :
                image = Image.open(f'image_Y{latest.lat}X{latest.lon}.jpg')
                # Remove the black on older panoramas and make image smaller height
                # remove added black margins
                # remove bar (if needed) and resize image to a height of 2016 or less
                '''
                image = np.array(image)
                if np.sum(image[:,-1]) == 0:
                    print("bars")
                    image[np.sum(a, axis=0)!=0]
                    image[np.sum(a, axis=1)!=0]
                
                resize = max(image.shape[0]//2016,1)
                image = image[::resize,::resize]
                image = Image.fromarray(image)
                '''
                image = trim(image)
                image.thumbnail([sys.maxsize, 2048], Image.LANCZOS)
                image.save(f'eval/image_Y{latest.lat}X{latest.lon}.jpg', "jpeg")
                #img.resize((new_width, new_height), Image.ANTIALIAS)
                
                # Image analysis for dist and bearing
                results = model(image)
                for r in results:
                    if r.masks != None :
                        '''
                        annotated = r.plot() # boxes=False
                        plt.imshow(annotated)
                        plt.axis('off')
                        plt.show()
                        '''
                        #print(len(r.masks.xy))
                        for i in range(len(r.masks.xy)): #r.masks.xy.size
                            # pole position
                            pole_pos_x = (r.boxes.xywh[i][0]+r.boxes.xywh[i][2]/2).cpu().numpy()
                            #heading + image_pc*360 goes right +180
                            #heading - image_pc*360 goes left -180
                            # clockwise angle
                            pole_offset = (pole_pos_x - (image.width/2)) / (image.width/2) * 180
                            bearing = (latest.heading + pole_offset) % 360
                            # obtain pole width pourcentage
                            editArr = np.copy(r.masks.xy[i])
                            pole_pos_y = (r.boxes.xywh[i][1]+r.boxes.xywh[i][3]/2).cpu().numpy()
                            pole_base_y = (r.boxes.xywh[i][1]+r.boxes.xywh[i][3]).cpu().numpy()
                            pole_width = 0 # (r.boxes.xywh[i][2]).cpu().numpy()/image.width
                            #divider = 0
                            '''
                            for i in range(3) :                             #pole_pos_y
                                lineArr = editArr[ np.absolute(editArr[:,1]-((pole_pos_y+pole_pos_y/2)/3*i)) < min((0.075*pole_pos_y),30) ]
                                if lineArr.shape[0] != 0:
                                    print(pole_width)
                                    pole_width += (lineArr[:,0].max() - lineArr[:,0].min()) #/image.width
                                    divider += 1
                            '''
                            lineArr = editArr[ np.absolute(editArr[:,1]-pole_pos_y) < min((0.075*pole_pos_y),30) ]
                            if lineArr.shape[0] != 0:
                                pole_width += (lineArr[:,0].max() - lineArr[:,0].min())/image.width
                                #divider += 1
                                # more precise method: get formula of function that passes through 2 point on the first side of pole b being X1 and fing its negative reciprocal function with the same b and check the Y that gives an X on the function or closest to it check smallest difference between result and real value
                            #elif editArr.shape[0] != 0:
                            #    pole_width = (editArr[:,0].max() - editArr[:,0].min())/image.width
                            #    print(pole_width)
                            # use size for distance
                            if pole_width >= 0.00134277344 : # != 0
                                '''
                                0.006155
                                0.016333
                                print(pole_width)
                                pole_width /= divider
                                print(pole_width)
                                '''
                                #distance = 0.488034043*pole_width
                                distance = ((0.005459064 + (10621.8 - 0.7759064)/(1 + (pole_width/0.000005948766)**2.55251)) + (0.007759064 + (118621.8 - 0.007759064)/(1 + (pole_width/0.000005948766)**2.675251))) / 2
                                
                                lat2, lon2 = get_point_at_distance(latest.lat, latest.lon, distance, bearing)
                                # save to new csv file 
                                print([lat2,lon2])
                                csv_writer.writerow([lat2,lon2,pole_width,distance,latest.lat,latest.lon]) # ,f'Lat:{latest.lat},Lon:{latest.lon},Pitch:{latest.pitch},Roll:{latest.roll}'
                                if line_count%100 == 0 :
                                    out_file.flush()
            if line_count == 1000:
                break
            line_count += 1
            # Remove all ocurrences of ^$\n
        '''
        content = out_file.read()
        content_new = re.sub('^$\n', '', content, flags = re.M)
        out_file.write(content_new)
        '''
    print(f'Processed {line_count} lines.') # 92 180 positions 46 090
