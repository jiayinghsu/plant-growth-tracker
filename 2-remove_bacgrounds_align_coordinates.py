import cv2
import numpy as np
import os

# Load image
date = '7-12-2022'
directory = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date + '/prep'
out1 = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date + '/cropped'
out2 = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date + '/final'

filenames = []
for filename in os.listdir(directory):
    if filename != '.DS_Store':
        filenames.append(filename)

for filename in filenames:
    img = cv2.imread(os.path.join(directory, filename))
    print(img.shape)

    img = img[150:3500, 210:2900]

    blurred = cv2.blur(img, (3, 3))
    canny = cv2.Canny(blurred, 100, 200)

    # remove white dots from the background
    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(canny,127,255,0)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:,cv2.CC_STAT_AREA]

    canny_nobg = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 100:   #keep
            canny_nobg[labels == i + 1] = 255

    # extract coordinates of boundaries
    pts = np.argwhere(canny_nobg>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    print(y1,y2,x1,x2)

    # crop images
    cropped = img[y1:y2, x1:x2]

    # resize images
    cropped = cv2.resize(cropped, dsize=(2000, 2000))
    print(cropped.shape)

    # save images
    cv2.imwrite(os.path.join(out1, filename), cropped)

    # remove handwritings and edges
    cropped = cropped[130:1900, 350:1840]

    # show images
    # cv2.imshow("Image", cropped)
    # cv2.waitKey(0)

    # save images
    cv2.imwrite(os.path.join(out2, filename), cropped)
