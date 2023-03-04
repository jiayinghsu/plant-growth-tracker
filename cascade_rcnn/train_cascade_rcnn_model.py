# Train a cascade R-CNN model on your dataset of plant leaf images and save it as "cascade_rcnn_model.h5".

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import plot_model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset, compute_ap

# Define the configuration for the model
class PlantLeafConfig(Config):
    NAME = "plant_leaf"
    NUM_CLASSES = 1 + 1  # Background + plant leaf
    STEPS_PER_EPOCH = 131
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001


# Define the dataset class
class PlantLeafDataset(Dataset):
    def load_dataset(self, dataset_dir):
        self.add_class("dataset", 1, "plant_leaf")
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, image_id + '.txt')
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        ann_path = info['annotation']
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        mask = np.zeros([info['height'], info['width'], len(lines)], dtype=np.uint8)
        class_ids = np.zeros(len(lines), dtype=np.int32)
        for i, line in enumerate(lines):
            class_id, coords = line.split()
            class_ids[i] = 1  # Only one class (plant leaf)
            coords = list(map(int, coords.split(',')))
            x1, y1, x2, y2 = coords
            mask[y1:y2, x1:x2, i] = 1
        return mask, class_ids


# Load the dataset
dataset = PlantLeafDataset()
dataset.load_dataset('path/to/your/dataset')
dataset.prepare()

# Load the ResNet50 model
model = ResNet50(include_top=False, input_shape=(256, 256, 3))

# Load the Mask R-CNN model with cascade classifier
rcnn_model = MaskRCNN(mode='training', model_dir='./', config=PlantLeafConfig())

# Train the model
rcnn_model.train(dataset, model, epochs=20, layers='heads', learning_rate=PlantLeafConfig.LEARNING_RATE)

# Save the trained model
rcnn_model.keras_model.save('cascade_rcnn_model.h5')