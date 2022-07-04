import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu
import os

# Load image
date = '7-3-2022'
directory = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date + '/converted'
out = '/Users/jiayingxu/Dropbox/Jiaying/data/' + date + '/prep'

for filename in os.listdir(directory):
    img = cv2.imread(os.path.join(directory, filename))

    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

    def filter_image(image, mask):
        r = image[:,:,0] * mask
        g = image[:,:,1] * mask
        b = image[:,:,2] * mask
        return np.dstack([r,g,b])

    thresh = threshold_otsu(img_gray)
    img_otsu  = img_gray < thresh
    filtered = filter_image(img, img_otsu)

    # show images
    cv2.imshow("Image", img_gray)
    cv2.waitKey(0)

    # save images
    cv2.imwrite(os.path.join(out, filename), filtered)
