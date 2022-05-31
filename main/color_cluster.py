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
def color_cluster(img, name='', save_dir='', debug=False):
    
    line_imgs = []
    
    if debug == True:
        # display original image
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5)) 
        plt.imshow(display, 'gray')
        plt.title('image before cluster')  
        plt.xticks([]),plt.yticks([])  
        plt.show()

    data = img.reshape((-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    flags = cv2.KMEANS_RANDOM_CENTERS
    NUM_CLUSTERS = 16

    compactness, labels, centers = cv2.kmeans(data, NUM_CLUSTERS, None, criteria, 10, flags)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    
    # get the class with the most pixels (white background)
    clusters = np.zeros([16], dtype=np.int32)
    for i in range(len(labels)):
        clusters[labels[i]] += 1
    center_bg = np.argmax(clusters)
    
    # TODO: how to deal with the generated images
    if debug == True:
        plt.subplots(figsize=(30, 20)) 

    for i in range(len(centers)):
        line_labels = np.full_like(labels, center_bg)
        line_labels[labels==i] = i
        
        line_res = centers[line_labels.flatten()]
        line_img = line_res.reshape((img.shape))

        if name != '':
            image_save_dir = os.path.join(save_dir, '.'.join(name.split(".")[:-1])+'_'+str(i)+'.png')
#             image_save_dir = os.path.join(save_dir, name+'_'+str(i)+'.png')
            cv2.imwrite(image_save_dir, line_img)
        
        if debug == True:
            line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
            line_imgs.append(line_img)

            plt.subplot(4,4,i+1)
            plt.imshow(line_img, 'gray')
            plt.title(str(i+1))
#             plt.title(str(centers[i]))  
            plt.xticks([]),plt.yticks([])  

    if debug == True:
        plt.show()
    return line_imgs