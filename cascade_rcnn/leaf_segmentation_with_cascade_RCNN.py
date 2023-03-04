# Segment plant leaves using cascade R-CNN and calculate the area of each segmented instance. 

import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cascade_rcnn_model.h5')

# Load the image to be segmented
image = cv2.imread('plant_leaf_image.jpg')

# Convert the image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image
image = cv2.resize(image, (800, 800))

# Perform segmentation using the model
results = model.detect([image])

# Get the number of instances segmented
num_instances = len(results[0]['rois'])

# Loop through the segmented instances and calculate their areas
for i in range(num_instances):
    # Get the bounding box coordinates for the instance
    y1, x1, y2, x2 = results[0]['rois'][i]
    # Crop the instance from the image
    instance = image[y1:y2, x1:x2]
    # Calculate the area of the instance
    area = np.count_nonzero(cv2.cvtColor(instance, cv2.COLOR_RGB2GRAY))
    # Print the area of the instance
    print('Instance {}: Area = {} pixels'.format(i+1, area))




##############

# Load cascade R-CNN model for plant leaves segmentation
model = cv2.dnn.readNetFromTensorflow('cascade_rcnn.pb')

# Load image
img = cv2.imread('plant_leaves.jpg')

# Set threshold for detection
confThreshold = 0.5

# Run inference on the image using the cascade R-CNN model
model.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
output = model.forward()

# Loop through all detected instances
for i in range(output.shape[2]):
    confidence = output[0, 0, i, 2]

    # Only consider instances with confidence above the threshold
    if confidence > confThreshold:
        classID = int(output[0, 0, i, 1])
        bbox = output[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        bbox = bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Calculate area of each instance
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print('Instance', i+1, 'Area:', area)

# Save results as CSV file
results = pd.DataFrame({'Instance': list(range(1, output.shape[2]+1)), 'Area': [int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for bbox in bboxes]})
results.to_csv('plant_leaves_results.csv', index=False)