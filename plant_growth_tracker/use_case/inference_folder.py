import os
import argparse
import cv2
import numpy as np
import random
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def sample_points_from_image(image, num_points):
    """
    Generate a set of random prompt points sampled uniformly
    over the full image.
    Each point is in the form [[x, y]].
    """
    h, w = image.shape[:2]
    points = []
    for _ in range(num_points):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        points.append([[x, y]])
    return np.array(points)

def overlay_masks(image, seg_map):
    """
    Overlay segmentation results on the image.
    
    - image: RGB numpy array.
    - seg_map: 2D numpy array with integer labels (0 = background, >0 = instance IDs).
    
    Each instance gets a random color and is labeled with its ID.
    """
    overlay = image.copy()
    unique_labels = np.unique(seg_map)
    # Build a dictionary mapping label -> random color.
    colors = {label: [random.randint(0, 255) for _ in range(3)] 
              for label in unique_labels if label != 0}
    
    for label in unique_labels:
        if label == 0:
            continue
        # Create a binary mask for this instance.
        mask = (seg_map == label).astype(np.uint8) * 255
        
        # Create a color overlay for this mask.
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[:] = colors[label]
        # Blend the colored mask with the image.
        overlay = cv2.addWeighted(overlay, 1.0,
                                  cv2.bitwise_and(colored_mask, colored_mask, mask=mask),
                                  0.5, 0)
        # Find contours to compute a centroid for drawing the label.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the largest contour.
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a folder of images to segment plant leaves and overlay the masks."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to the folder containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Folder to save output images with overlayed masks.")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                        help="Path to the SAM2 checkpoint file (e.g., sam2_hiera_small.pt).")
    parser.add_argument("--model_cfg", type=str, required=True,
                        help="Path to the SAM2 model config YAML (e.g., sam2_hiera_s.yaml).")
    parser.add_argument("--fine_tuned_model_weights", type=str, required=True,
                        help="Path to the fine-tuned model weights file (e.g., fine_tuned_sam2_1000.torch).")
    parser.add_argument("--num_points", type=int, default=30,
                        help="Number of prompt points to sample per image.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference (e.g., cuda or cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build the SAM2 model and load the fine-tuned weights.
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(
        torch.load(args.fine_tuned_model_weights, map_location=args.device, weights_only=True)
    )

    # Process all images in the input folder.
    for filename in os.listdir(args.images_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(args.images_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            continue
        # Convert from BGR to RGB.
        img = img[..., ::-1]

        # Resize image so the larger dimension is 1024.
        scale = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        resized_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        # Generate prompt points from the entire image.
        input_points = sample_points_from_image(resized_img, args.num_points)
        point_labels = np.ones((input_points.shape[0], 1))
        
        with torch.no_grad():
            predictor.set_image(resized_img)
            # Obtain candidate masks.
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=point_labels
            )

        # Process predicted masks.
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0]
        sorted_indices = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_indices]

        # Create a segmentation map from the predicted masks.
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
        for i in range(sorted_masks.shape[0]):
            mask_i = sorted_masks[i]
            if (mask_i * occupancy_mask).sum() / (mask_i.sum() + 1e-6) > 0.15:
                continue
            mask_bool = mask_i.astype(bool)
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = i + 1
            occupancy_mask[mask_bool] = True

        # Overlay the segmentation on the image.
        overlay_img = overlay_masks(resized_img, seg_map)
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(out_path, overlay_bgr)
        print(f"Processed {filename} -> {out_path}")

if __name__ == "__main__":
    main()
