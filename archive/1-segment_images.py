import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu
import os

# Load image
date = '00-00-0000'
directory = '~/data/' + date + '/converted'
out = '~/data/' + date + '/prep'

filenames = []
for filename in os.listdir(directory):
    if filename != '.DS_Store':
        filenames.append(filename)

for filename in filenames:
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
    
    # save images
    cv2.imwrite(os.path.join(out, filename), filtered)



