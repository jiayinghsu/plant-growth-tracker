import cv2
import imutils
import imutils.perspective as persp
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load image
date = '00-00-0000'
directory = '~/data/' + date + '/final'
out = '~/data/' + date

d = {}
filenames = []
for filename in os.listdir(directory):
    if filename != '.DS_Store':
        filenames.append(filename)

for filename in filenames:
    img = cv2.imread(os.path.join(directory, filename))

    # convert into gray scale for detecting the contours i.e. outlines of each individual object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove the excessive noise using THRESHOLD
    thresh, thresh_img = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # show the image in THRESH
    rgb_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2RGB)

    # find the total contours
    contours, _ = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    centres = []
    areas = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2000:
            continue
        moments = cv2.moments(contours[i])
        centres.append((int(moments['m10']/(moments['m00']+0.1)), int(moments['m01']/(moments['m00']+0.1))))
        areas.append(area)
        cv2.circle(gray, centres[-1], 3, (255, 255, 255), -1)
        cv2.drawContours(gray, contours, -1, (255, 255, 255), 2)

    areas.sort()
    areas_sel = areas[-4:]

    if len(areas_sel) == 1:
        areas_sel.extend([0,0,0])
    elif len(areas_sel) == 2:
        areas_sel.extend([0, 0])
    elif len(areas_sel) == 3:
        areas_sel.append(0)

    areas_sel.append(sum(areas_sel) / len(areas_sel))
    d[filename] = areas_sel

df = pd.DataFrame(d)
df = df.T
df.rename(columns={0:1, 1:2, 2:3, 3:4, 4:"Average"}, inplace=True)
df['Date'] = date
print(df)

df.to_csv(os.path.join(out, date+'.csv'))
