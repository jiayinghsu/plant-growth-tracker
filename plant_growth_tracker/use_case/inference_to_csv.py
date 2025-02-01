import os
import argparse
import csv
import cv2
import numpy as np
import random
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def sample_points_from_image(image, num_points):
    """
    Randomly sample prompt points over the entire image.
    Each point is in the form [[x, y]].
    """
    h, w = image.shape[:2]
    points = []
    for _ in range(num_points):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        points.append([[x, y]])
    return np.array(points)

def build_segmentation_map(predictor, resized_img, num_points):
    """
    Given the predictor and a resized image, sample prompt points and use the model
    to produce a segmentation map. Returns a segmentation map (uint8 array) with instance labels.
    """
    input_points = sample_points_from_image(resized_img, num_points)
    point_labels = np.ones((input_points.shape[0], 1))
    
    with torch.no_grad():
        predictor.set_image(resized_img)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=point_labels
        )

    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_indices = np.argsort(np_scores)[::-1]
    sorted_masks = np_masks[sorted_indices]

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

    return seg_map

def compute_instance_properties(seg_map, instance_id):
    """
    Given a segmentation map and an instance ID, compute the area (number of pixels)
    and the centroid (x, y) of that instance.
    Returns (area, centroid_x, centroid_y).
    """
    instance_mask = (seg_map == instance_id).astype(np.uint8) * 255
    area = int(np.count_nonzero(instance_mask))
    
    contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroid_x, centroid_y = 0.0, 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            centroid_x = float(M["m10"] / M["m00"])
            centroid_y = float(M["m01"] / M["m00"])
    return area, centroid_x, centroid_y

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a folder of images using the fine-tuned SAM2 model and output a CSV summary."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to the folder containing input images.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to the output CSV file.")
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

    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(
        torch.load(args.fine_tuned_model_weights, map_location=args.device, weights_only=True)
    )

    csv_rows = []  # List to store CSV rows
    image_files = [f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("No image files found in", args.images_dir)
        return

    for filename in image_files:
        image_path = os.path.join(args.images_dir, filename)
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            print(f"Could not read image: {image_path}")
            continue
        orig_img = orig_img[..., ::-1]
        orig_h, orig_w = orig_img.shape[:2]

        scale = np.min([1024 / orig_w, 1024 / orig_h])
        resized_img = cv2.resize(orig_img, (int(orig_w * scale), int(orig_h * scale)))
        seg_map_resized = build_segmentation_map(predictor, resized_img, args.num_points)
        seg_map = cv2.resize(seg_map_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        instance_labels = np.unique(seg_map)
        image_instance_id = 1
        for inst in instance_labels:
            if inst == 0:
                continue
            area, cx, cy = compute_instance_properties(seg_map, inst)
            csv_rows.append({
                "Object ID": image_instance_id,
                "Image origin": filename,
                "Class": "leaf",
                "Area": area,
                "X Position": cx,
                "Y Position": cy
            })
            image_instance_id += 1

        print(f"Processed {filename}: found {image_instance_id - 1} leaf objects.")

    header = ["Object ID", "Image origin", "Class", "Area", "X Position", "Y Position"]
    with open(args.output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Inference complete. CSV saved to {args.output_csv}")

if __name__ == "__main__":
    main()