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

from streetview import search_panoramas, get_panorama
from math import asin, atan2, cos, degrees, radians, sin
from ultralytics import YOLO
from os.path import exists
from PIL import Image
import numpy as np
import csv
import cv2

model = YOLO('conv_dataset/runs/segment/train26/weights/best.pt')  # load a pretrained model (use best.pt)

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

lat = 52.20472
lon = 0.14056
distance = 15
bearing = 90
lat2, lon2 = get_point_at_distance(lat, lon, distance, bearing)

with open('CheckPoints.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    with open('PolePoints_.csv', 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',')
        csv_writer.writerow(['Y','X'])
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else : #every odd line #test case if line_count == 1 
                #print(f'\t coords: Y: {row[0]} X: {row[1]}')
                panos = search_panoramas(lat=row[0], lon=row[1])
                latest = panos[-1]
                print(latest)
                image = None
                '''
                if not exists(f'image_Y{latest.lat}X{latest.lon}.jpg') :
                    image = get_panorama(pano_id=latest.pano_id)
                    image.save(f'image_Y{latest.lat}X{latest.lon}.jpg', "jpeg")
                else:
                    image = Image.open(f'image_Y{latest.lat}X{latest.lon}.jpg')
                '''
                if exists(f'image_Y{latest.lat}X{latest.lon}.jpg') :
                    image = Image.open(f'image_Y{latest.lat}X{latest.lon}.jpg')
                    # Remove the black on older panoramas and make image smaller height
                    # remove added black margins
                    # remove bar (if needed) and resize image to a height of 2016 or less
                    
                    image = np.array(image)
                    if np.sum(image[:,-1]) == 0:
                        print("bars")
                        image[np.sum(a, axis=0)!=0]
                        image[np.sum(a, axis=1)!=0]
                    
                    resize = max(image.shape[0]//2016,1)
                    image = image[::resize,::resize]
                    image = Image.fromarray(image)
                    # Image analysis for dist and bearing
                    results = model(image)
                    for r in results:
                        if r.masks != None :
                            # pole position
                            pole_pos_x = (r.boxes.xywh[0][0]).cpu().numpy() #+r.boxes.xywh[0][2]/2
                            #heading + image_pc*360 goes right +180
                            #heading - image_pc*360 goes left -180
                            # clockwise angle
                            pole_offset = (pole_pos_x - (image.width/2)) / (image.width/2) * 180
                            bearing = (latest.heading + pole_offset) % 360
                            # obtain pole width pourcentage
                            editArr = np.copy(r.masks.xy[0])
                            pole_pos_y = (r.boxes.xywh[0][1]+r.boxes.xywh[0][3]/2).cpu().numpy()
                            lineArr = editArr[ np.absolute(editArr[:,1]-pole_pos_y) < 30 ]
                            pole_width = (lineArr[:,0].max() - lineArr[:,0].min())/image.width
                            # use size for distance 0.0064/0.0143432617  dist in km / pourcent of image of width
                            distance = 0.446202554*pole_width
                            lat2, lon2 = get_point_at_distance(latest.lat, latest.lon, distance, bearing)
                            # save to new csv file 
                            print([lat2,lon2])
                            csv_writer.writerow([lat2,lon2])
                            out_file.flush()
            line_count += 1
            # Remove all ocurrences of ^$\n
    print(f'Processed {line_count} lines.') # 92 180 positions 46 090
'''
lat1 = <latitude value of point A>
lon1 = <longitude value of point A>
lat2 = <latitude value of point B>
lon2 = <longitude value of point B>

r = 6371 #radius of Earth (KM)
p = 0.017453292519943295  #Pi/180
a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p)*math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p)) / 2

d = 2 * r * math.asin(math.sqrt(a)) #2*R*asin
'''
'''
78+80+82+84+86  /31
+89+91+94+97+99+102+104+107+109+112+114+117+119+122+124+127+130+132+135+137+140+142+145+147+150+152 /26
average circumference with rotten poles = 114.42
average diameter with rotten poles = 36.42
average circumference without rotten poles = 120.65
average diameter without rotten poles = 38.40

based on data from this document: https://www.hydroquebec.com/data/location-structures/docs/norme-uc-structure-poteau.pdf
0.0143432617
21 235px     37.42   
360          641.49 cm 6.42 m  7179px 0.0064/0.0143432617

'''