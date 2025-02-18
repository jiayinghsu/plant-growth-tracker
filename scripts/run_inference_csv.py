import os
import glob
import csv
import argparse
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Inference to CSV for single-class model.")
    parser.add_argument("--task", choices=["leaf", "plant"], default="leaf",
                        help="Which single class the model was trained on.")
    parser.add_argument("--weights-file", required=True,
                        help="Path to your trained .pth model weights.")
    parser.add_argument("--input-folder", required=True,
                        help="Folder containing images for inference.")
    parser.add_argument("--output-csv", required=True,
                        help="Path to the CSV file to write results.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Confidence threshold.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Single class name
    class_name = args.task  # "leaf" or "plant"

    # 1) Build config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = args.weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # single class

    # 2) Create predictor
    predictor = DefaultPredictor(cfg)

    # 3) Gather all images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(args.input_folder, e)))
    image_paths.sort()

    # 4) Prepare CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Object ID", "Image origin", "Class", "Area", "X Position", "Y Position"])

        # 5) Run inference per image
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: could not open {img_path}. Skipping.")
                continue

            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")

            # If no instances, skip or continue
            if len(instances) == 0:
                print(f"No instances found in {img_name}.")
                continue

            # pred_masks is a BoolTensor of shape [N, H, W]
            pred_masks = instances.pred_masks if instances.has("pred_masks") else []
            # pred_classes = instances.pred_classes  # all 0 if single class

            for idx, mask_tensor in enumerate(pred_masks):
                mask = mask_tensor.numpy()  # convert to numpy bool array

                area = float(mask.sum())  # count True pixels

                # Centroid
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    cx, cy = 0.0, 0.0
                else:
                    cx = float(xs.mean())
                    cy = float(ys.mean())

                # Write row to CSV
                writer.writerow([
                    idx + 1,         # object ID (1-based per image)
                    img_name,        # image name
                    class_name,      # single class
                    area,            # pixel area
                    cx,              # centroid X
                    cy               # centroid Y
                ])

    print(f"Done! Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()