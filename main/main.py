import cv2
import numpy as np
import os, random
from copy import deepcopy
import csv, json, boto3
import math
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

import imutils
import numpy as np

from matplotlib import rcParams
from text_Rekognition import detectText
from axes_detection import detectAxes
from color_cluster import color_cluster
from Data_Extraction import data_of_one_line
#pre processing
img_dir = './test/ori/'

cluster_img_save = False
if cluster_img_save:
    cluster_save_dir = './test/4-ColorCluster/'
    if not os.path.isdir(cluster_save_dir):
        os.mkdir(cluster_save_dir)


img_text = {}
image_text = {}
client = boto3.client('rekognition', region_name = 'us-west-2')

debug = True

for index, path in enumerate(Path(img_dir).iterdir()):
    if path.name.lower().endswith('.png') or path.name.lower().endswith('.jpg') or path.name.lower().endswith('.jpeg'):
        filepath = img_dir + "/" + path.name

        print("[{0}] file name: {1}".format(index, path.name))
        
        ##### Axis Detection #####
        xaxis, yaxis, x_up_axis, y_right_axis = detectAxes(filepath)

        xaxis_yvalue = xaxis[1]-1
        yaxis_xvalue = yaxis[0]+1
        up_yvalue = x_up_axis[1]+1
        right_xvalue = y_right_axis[0]-1


        ##### Text Rekognition #####
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = detectText(path, image, image_text, img_text)
        image = detectText(path, image, image_text, img_text)
        
        image_crop_del = image[x_up_axis[1]+1:xaxis[1], yaxis[0]+1:y_right_axis[0]]

        
        ##### X and Y Coords #####
        if debug:
            image_show = deepcopy(image)

        x_coords = []
        y_coords = []
        
        # judge if the detected text is int outside the axis
        for i in range(len(img_text[path.name])):
            if img_text[path.name][i][0].replace('.','',1).isdigit():
                if img_text[path.name][i][1][0] + img_text[path.name][i][1][2] < yaxis_xvalue:
        #             print(img_text[path.name][i][0], "y coords", img_text[path.name][i][1])

                    axis_text = img_text[path.name][i][0]
                    bbox = img_text[path.name][i][1]
                    y_mid = bbox[1] + bbox[3]/2
                    y_coords.append([float(axis_text), bbox, y_mid])

                    if debug:
                        y = int(img_text[path.name][i][1][1] + img_text[path.name][i][1][3]/2)
                        cv2.line(image_show, (0, y), (image_show.shape[1], y),  (0, 0, 255), 1)
                if img_text[path.name][i][1][1] > xaxis_yvalue:
        #             print(img_text[path.name][i][0], "x coords", img_text[path.name][i][1])

                    axis_text = img_text[path.name][i][0]
                    bbox = img_text[path.name][i][1]
                    x_mid = bbox[0] + bbox[2]/2
                    x_coords.append([int(axis_text), bbox, x_mid])

                    if debug:
                        x = int(bbox[0] + bbox[2]/2)
                        cv2.line(image_show, (x, 0), (x, image_show.shape[0]),  (0, 0, 255), 1)


        # display image
        if debug:
            plt.imshow(image_show, 'gray')
            plt.title('axis visualisation')  
            plt.xticks([]),plt.yticks([])  
            plt.show()

#             plt.imshow(image_crop_del, 'gray')
#             plt.title('image cropped and delete text')  
#             plt.xticks([]),plt.yticks([])  
#             plt.show()

        ##### Color Cluster #####
        image_crop_del = cv2.cvtColor(image_crop_del, cv2.COLOR_RGB2BGR)
        if cluster_img_save:
            line_imgs = color_cluster(image_crop_del, path.name, cluster_save_dir, debug=True)
        else:
            line_imgs = color_cluster(image_crop_del, debug=True)

# all 16 images
line_img_indices = [i+1 for i in range(16)]

completeness = []
for line_img_i in line_img_indices:
    image = line_imgs[line_img_i-1]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    data_points = data_of_one_line(image, debug=True)
    num_of_points = len(data_points)
    completeness.append(num_of_points)

select_img_index = []
for i in range(7):
    max_index = completeness.index(max(completeness))
    select_img_index.append(max_index)
    completeness[max_index] = -1

data_save_dir = "./test/DataResult/"
if not os.path.isdir(data_save_dir):
    os.mkdir(data_save_dir)

            
for line_img_i in select_img_index:

    image = line_imgs[line_img_i]
    print(image.shape[1])
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # data points in image pixels
    data_points = data_of_one_line(image, debug=True)
    print(len(data_points))

    # below: convert points in pixels to real line chart data
    data_points = np.array(data_points)
    data_points[:,0] += yaxis_xvalue
    data_points[:,1] += up_yvalue
    sort_index = np.lexsort((data_points[:,1], data_points[:,0])) # sort the data points according to x values
    data_points = data_points[sort_index,:]

    y_value_pixels = np.zeros((len(y_coords), 2))
    for i in range(len(y_coords)):
        y_value_pixels[i] = [y_coords[i][0], y_coords[i][2]]

    sort_index = np.lexsort((y_value_pixels[:,0], y_value_pixels[:,1])) # sort the data points according to x values
    y_value_pixels = y_value_pixels[sort_index,:]

    
    data_file = os.path.join(data_save_dir + str(line_img_i) + ".csv")
    assert(not os.path.isfile(data_file))

    with open(data_file,'w',newline='') as f:
        writer=csv.writer(f)
    #     header="img_name","x","y"
        header="x","y"
        writer.writerow(header)



        # x precision 1

        for i in range(len(x_coords)-1):
            interval_len = x_coords[i+1][0] - x_coords[i][0]
            interval_len_pixel_f = x_coords[i+1][2] - x_coords[i][2]

            x_results = [x for x in range(x_coords[i][0], x_coords[i+1][0])]
            x_pixel_floats = [x_coords[i][2] + interval_len_pixel_f / interval_len * (x-x_coords[i][0]) for x in range(x_coords[i][0], x_coords[i+1][0])]

            indices = np.searchsorted(data_points[:,0], x_pixel_floats)

            for xi in range(len(x_results)): 
                if x_pixel_floats[xi] > data_points[0,0] and x_pixel_floats[xi] < data_points[-1,0]:
                    if data_points[indices[xi],0] == x_pixel_floats[xi]:
                        y_pixel = data_points[indices[xi],1]
                    else:
                        y_pixel = ( (data_points[indices[xi],1] - data_points[indices[xi]-1,1]) / 
                                    (data_points[indices[xi],0] - data_points[indices[xi]-1,0]) * 
                                    (x_pixel_floats[xi] - data_points[indices[xi]-1,0]) ) + data_points[indices[xi]-1,1]

                    if y_pixel > min(y_value_pixels[:,1]) and y_pixel < max(y_value_pixels[:,1]):
                        y_insert_index = np.searchsorted(y_value_pixels[:,1], y_pixel)

                        y_result = ((y_value_pixels[y_insert_index,0] - y_value_pixels[y_insert_index-1,0]) / 
                                    (y_value_pixels[y_insert_index,1] - y_value_pixels[y_insert_index-1,1]) * 
                                    (y_pixel - y_value_pixels[y_insert_index-1,1])) + y_value_pixels[y_insert_index-1,0]

                        xy_result = [x_results[xi], y_result]
                        writer.writerow(xy_result)
