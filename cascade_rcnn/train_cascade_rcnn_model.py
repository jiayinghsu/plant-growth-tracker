'''
Here, we first load the pretrained model 'cascade_rcnn_model.h5' and then load and preprocess the input image. We then perform segmentation on the image using the model and extract the segmentation results, including the masks, class IDs, and scores.

Next, we calculate the area of each segmented instance by finding the contour of the mask and using the cv2.contourArea function. We store the areas in a list.

Finally, we save the results as a CSV file using the pandas library. We create a dictionary of the class IDs, scores, and areas, and then convert it to a pandas DataFrame. We then use the to_csv function to save the DataFrame as a CSV file.
'''

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

# Load the pretrained model
model = load_model('cascade_rcnn_model.h5', compile=False)

# Load the image and preprocess it
image = cv2.imread('plant_leaves.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (800, 800))
image = image.astype(np.float32) / 255.0

# Perform segmentation on the image
results = model.detect([image])

# Extract the segmentation results
r = results[0]
masks = r['masks']
class_ids = r['class_ids']
scores = r['scores']

# Calculate the area of each segmented instance
areas = []
for i in range(masks.shape[2]):
    mask = masks[:, :, i]
    contour = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    area = cv2.contourArea(contour)
    areas.append(area)

# Save the results as a CSV file
data = {'Class ID': class_ids, 'Score': scores, 'Area': areas}
df = pd.DataFrame(data)
df.to_csv('segmentation_results.csv', index=False)