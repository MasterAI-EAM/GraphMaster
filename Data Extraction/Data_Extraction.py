'''extract data for image with just one line'''

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

def data_of_one_line(img, name='', save_dir='', data_file='', debug=False):
    
    data_points = []
    
    if debug==True:
        plt.subplots(figsize=(15, 10)) 

        # display image
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,2,1)
        plt.imshow(display, 'gray')
        plt.title('original line image')  
        plt.xticks([]),plt.yticks([])  

    # use color cluster again to distinguish line and background
    data = img.reshape((-1,3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    NUM_CLUSTERS = 2
    compactness, labels, centers = cv2.kmeans(data, NUM_CLUSTERS, None, criteria, 10, flags)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    
    # get the class with the most pixels (white background)
    clusters = np.zeros([16], dtype=np.int32)
    for i in range(len(labels)):
        clusters[labels[i]] += 1
    center_bg = np.argmax(clusters)
    
    for i in range(len(centers)):
        line_labels = np.full_like(labels, center_bg)
        line_labels[labels==i] = i
        
        if i == center_bg: # disgard the background class
            continue
        
        img_h, img_w = img.shape[0],img.shape[1]
        labels_img = labels.reshape((img_h, img_w))
        
        data_pts = np.full((img_w, 2), -1)
        last_data_pt = [-1,-1]
        max_h_diff = 0
        for x in range(3, img_w): # ignore some margin in left
            ver_labels = labels_img[:,x].reshape(-1)
            if np.argwhere(ver_labels!=center_bg).shape[0] != 0:
                
                ver_line_pixels = np.argwhere(ver_labels!=center_bg).reshape(-1,1)
                
                # clustering on one vertical line to get the position of line data
                MAX_GAP=3
                clustering = DBSCAN(eps=MAX_GAP, min_samples=1).fit(ver_line_pixels)
                ver_centers = []
                ver_counts = []
                for c in np.unique(clustering.labels_):
                    pts = ver_line_pixels[np.argwhere(clustering.labels_==c)].reshape(-1)
                    ver_centers.append(np.mean(pts, dtype=np.int32))
                    ver_counts.append(len(pts))
                
                ver_centers = np.array(ver_centers)
                ver_counts = np.array(ver_counts)
                if ver_centers.size == 1:
                    if last_data_pt[0] != -1 and abs(ver_centers[0]-last_data_pt[1])>max_h_diff*2: 
                        data_h = -1
                    else:
                        data_h = ver_centers[0]
                elif last_data_pt[0] != -1:
                    nearest_id = (np.abs(ver_centers - last_data_pt[1])).argmin()
                    thickest_id = ver_counts.argmax()
                    if ver_counts[nearest_id] < 3 and ver_counts[thickest_id] >= 3:
                        data_h = ver_centers[thickest_id]
                    else:
                        data_h = ver_centers[nearest_id]
                else:
                    data_h = random.choice(ver_centers) 
                    
                if data_h != -1:
                    display = cv2.circle(img, (x, data_h), 1, (255,0,255), 2)
                    data_points.append((x, data_h))

                    data_pts[x] = [x, data_h]
                    max_h_diff = max(data_h - last_data_pt[1], max_h_diff)
                    last_data_pt = [x, data_h]
              
        if data_file != '':  
            data_to_save = np.full(((img_w-1)//10+1, 2), -1)
            for x in range(data_to_save.shape[0]):
                data_to_save[x] = [x*10, data_pts[x*10,1]]
                
            row_to_save = [name]
            row_to_save.append(data_to_save)
            with open(data_file,'a',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(row_to_save)
            
        if name != '':
            image_save_dir = os.path.join(save_dir, name)
            cv2.imwrite(image_save_dir, display)
        
        if debug==True:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            plt.subplot(1,2,2)
            plt.imshow(display, 'gray')
            plt.title('data points shown in dot')  
            plt.xticks([]),plt.yticks([])  
    if debug==True:
        plt.show()
    
    return data_points