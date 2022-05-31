'''
Code for this part from 
https://github.com/Cvrane/ChartReader/blob/master/code/AWS-Text-Rekognition.ipynb
'''

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
from main import client
def expand(points, margin = 1):
    return np.array([
        [[points[0][0][0] - margin, points[0][0][1] - margin]],
        [[points[1][0][0] + margin, points[1][0][1] - margin]],
        [[points[2][0][0] + margin, points[2][0][1] + margin]],
        [[points[3][0][0] - margin, points[3][0][1] + margin]]])

def detectText(path, image, image_text, img_text):
    
    img_height, img_width, channels = image.shape
    _, im_buf = cv2.imencode("." + path.name.split(".")[-1], image)
        
    response = client.detect_text(
        Image = {
            "Bytes" : im_buf.tobytes()
        }
    )
    
    if path.name not in image_text:
        image_text[path.name] = {}
        image_text[path.name]['TextDetections'] = response['TextDetections']
    else:
        image_text[path.name]['TextDetections'].extend(response['TextDetections'])
        
    textDetections = response['TextDetections']
        
    if path.name not in img_text:
        img_text[path.name] = []
            
    for text in textDetections:
        if text['Type'] == 'WORD' and text['Confidence'] >= 80:
                
            vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
            vertices = np.array(vertices, np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            
            image = cv2.fillPoly(image, [expand(vertices)], (255, 255, 255))
                  
            left = np.amin(vertices, axis=0)[0][0]
            top = np.amin(vertices, axis=0)[0][1]
            right = np.amax(vertices, axis=0)[0][0]
            bottom = np.amax(vertices, axis=0)[0][1]
            
            img_text[path.name].append(
                (
                    text['DetectedText'],
                    (
                        int(left),
                        int(top),
                        int(right - left),
                        int(bottom - top)
                    )
                )
            )

    return image