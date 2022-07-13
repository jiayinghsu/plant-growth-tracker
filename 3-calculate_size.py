import cv2
import imutils
import imutils.perspective as persp
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load image
date = '7-12-2022'
directory = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date + '/final'
out = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date

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

    # # find the total contours
    # conts = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('Total # of contours by CV2: ', len(conts))
    #
    # # fine tune contours returned by cv2 using imutils
    # conts = imutils.grab_contours(conts)
    # print('Total # oc Contours by imUTILS: ', len(conts))
    #
    #
    # # create the empy image with the original dimension
    # cont_img = np.zeros(img.shape)
    #
    # # draw the contours in the empty image i.e. cont_img
    # cont_img = cv2.drawContours(cont_img, conts, -1, (0,255,0), 2)

    # show the image with contours i.e. only outline (borders)
    #
    # def midPoint(ptA, ptB):
    #     return ((ptA[0]+ptB[0])/2, (ptA[1]+ptB[1])/2)
    #
    # # loop over all the contour coordinates
    # for c in conts:
    #     # extract box points
    #     box = cv2.minAreaRect(c)
    #     print(box)
    #     box = cv2.boxPoints(box)
    #     # convert the boxPoints into integer
    #     box = np.array(box, dtype='int')
    #
    #     if cv2.contourArea(c) < 500:
    #         continue
    #
    #     # draw the contour
    #     cv2.drawContours(cont_img, [c], -1, (0,255,0), 2)
    #     cv2.drawContours(cont_img, [box], -1, (255, 255, 255), 1)
    #
    #     print(box)
    #     for (x,y) in box:
    #         cv2.circle(cont_img, (x, y), 2, (255,0,0), 2)
    #         (tl, tr, br, bl) = box
    #
    #         # calcuclate midPoint dots for top and bottom
    #         (tlX, trX) = midPoint(tl, tr)
    #         (brX, blX) = midPoint(br, bl)
    #
    #         # draw the midpoint dots for top and bottom
    #         cv2.circle(cont_img, (int(tlX), int(trX)), 1, (255,0,0), 2)
    #         cv2.circle(cont_img, (int(brX), int(blX)), 1, (255, 0, 0), 2)
    #
    #         # connect the midpoint using line
    #         cv2.line(cont_img, (int(tlX), int(trX)), (int(brX), int(blX)), (255, 255, 255), 1)
    #
    #         # calculate the distance based on the midPoints
    #         dA = dist.euclidean((tlX, trX), (brX, blX))
    #
    #         # print the size in PX on each contour rectangle
    #         cv2.putText(cont_img, "{:.1f} px".format(dA), (int(tlX - 10), int(trX - 10)),
    #                     cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    #
    #         # calcuclate midPoint dots for left and right
    #         (tlX, trX) = midPoint(tl, bl)
    #         (brX, blX) = midPoint(tr, br)
    #
    #         # draw the midpoint dots for left and right
    #         cv2.circle(cont_img, (int(tlX), int(trX)), 1, (255, 0, 0), 2)
    #         cv2.circle(cont_img, (int(brX), int(blX)), 1, (255, 0, 0), 2)
    #
    #         # connect the midpoint using line
    #         cv2.line(cont_img, (int(tlX), int(trX)), (int(brX), int(blX)), (255, 255, 255), 1)
    #
    #         # calculate the distance based on the midPoints
    #         dB = dist.euclidean((tlX, trX), (brX, blX))
    #
    #         # print the size in PX on each contour rectangle
    #         cv2.putText(cont_img, "{:.1f} px".format(dB), (int(brX+10), int(blX+10)),
    #                     cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    #

    contours, _ = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    # print(len(contours))

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

    # # show the processed image with contours printed
    # cv2.imshow("002", gray)
    # cv2.waitKey(0)

# print(d)
df = pd.DataFrame(d)
df = df.T
df.rename(columns={0:1, 1:2, 2:3, 3:4, 4:"Average"}, inplace=True)
df['Date'] = date
print(df)

df.to_csv(os.path.join(out, date+'.csv'))
