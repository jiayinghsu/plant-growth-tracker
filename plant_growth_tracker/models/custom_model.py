import torch
from transformers import SamProcessor, SamModel
import numpy as np
from PIL import Image
import cv2

class CustomSAMModel:
    def __init__(self, model_path: str, processor_path: str, device: str = 'cpu'):
        """
        Initializes the custom SAM model.

        Args:
            model_path (str): Path to the trained model weights.
            processor_path (str): Path to the trained processor.
            device (str): 'cpu' or 'cuda' for GPU acceleration.
        """
        self.device = torch.device(device)
        self.model = SamModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.processor = SamProcessor.from_pretrained(processor_path)

    def predict(self, image):
        """
        Predicts segmentation masks for the given image.

        Args:
            image (PIL.Image.Image or np.ndarray): The input image.

        Returns:
            np.ndarray: The predicted mask.
        """
        # Convert image to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            image_pil = image

        # Preprocess the image
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
            pred_masks = outputs.pred_masks  # Shape: (batch_size, num_masks, H, W)

        # Post-process the masks
        pred_masks = torch.sigmoid(pred_masks)
        pred_masks = pred_masks.cpu().numpy()

        # Threshold the masks
        binary_masks = (pred_masks > 0.5).astype(np.uint8)

        return binary_masks  # Shape: (batch_size, num_masks, H, W)
