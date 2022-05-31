"This code gives a best estimate of the x and y axis (horizontal and vertical axes) for the plot/chart."
"Method to detect x and y axis"

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

def findMaxConsecutiveOnes(nums) -> int:
    count = maxCount = 0
    
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
        else:
            maxCount = max(count, maxCount)
            count = 0
                
    return max(count, maxCount)

def detectAxes(filepath, threshold=None, debug=False):
    if filepath is None:
        return None, None
    
    if threshold is None:
        threshold = 10
    
    image = cv2.imread(filepath)
    height, width, channels = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the max-consecutive-ones for eah column in the bw image, and...
    # pick the "first" index that fall in [max - threshold, max + threshold]
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[:, idx] < 200) for idx in range(width)]
    
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < width/2:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            start_idx += 1
            while abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
                maxindex = start_idx
                start_idx += 1
            break
            
        start_idx += 1
    
    yaxis = (maxindex, 0, maxindex, height)
    
    start_idx, y_right_index, maxcount = width-1, width-1, max(maxConsecutiveOnes)
    while start_idx > width/2:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            y_right_index = start_idx
            start_idx -= 1
            while abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
                y_right_index = start_idx
                start_idx -= 1
            break
            
        start_idx -= 1
           
    y_right_axis = (y_right_index, 0, y_right_index, height)
    
    if debug:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image)

        ax[1].plot(maxConsecutiveOnes, color = 'k')
        ax[1].axhline(y = max(maxConsecutiveOnes) - 10, color = 'r', linestyle = 'dashed')
        ax[1].axhline(y = max(maxConsecutiveOnes) + 10, color = 'r', linestyle = 'dashed')
        ax[1].vlines(x = maxindex, ymin = 0.0, ymax = maxConsecutiveOnes[maxindex], color = 'b', linewidth = 4)

        plt.show()

    # Get the max-consecutive-ones for eah row in the bw image, and...
    # pick the "last" index that fall in [max - threshold, max + threshold]
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[idx, :] < 200) for idx in range(height)]
    start_idx, maxindex, maxcount = height-1, height-1, max(maxConsecutiveOnes)
    while start_idx > height/2:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            start_idx -= 1
            while abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
                maxindex = start_idx
                start_idx -= 1
            break
            
        start_idx -= 1
            
    cv2.line(image, (0, maxindex), (width, maxindex),  (255, 0, 0), 2)
    xaxis = (0, maxindex, width, maxindex)
    
    start_idx, x_up_index, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < height/2:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            x_up_index = start_idx
            start_idx += 1
            while abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
                x_up_index = start_idx
                start_idx += 1
            break
            
        start_idx += 1
            
    x_up_axis = (0, x_up_index, width, x_up_index)
    
    if debug:
        rcParams['figure.figsize'] = 15, 8

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, aspect = 'auto')
                
    return xaxis, yaxis, x_up_axis, y_right_axis