import os
import glob
import argparse
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for single-class (leaf or plant).")
    parser.add_argument("--task", choices=["leaf", "plant"], default="leaf",
                        help="Which single class the model was trained on.")
    parser.add_argument("--weights-file", required=True,
                        help="Path to trained .pth model weights.")
    parser.add_argument("--input-folder", required=True,
                        help="Folder containing images for inference.")
    parser.add_argument("--output-folder", required=True,
                        help="Folder to save annotated images.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Confidence score threshold for predictions.")
    return parser.parse_args()

def main():
    args = parse_args()

    class_name = args.task  # "leaf" or "plant"
    thing_classes = [class_name]

    # 1) Build config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = args.weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class

    # 2) Create predictor
    predictor = DefaultPredictor(cfg)

    # 3) Prepare output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # 4) Gather all images from input folder
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(args.input_folder, e)))
    image_paths.sort()

    # 5) Run inference on each image
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not open {img_path}. Skipping.")
            continue

        outputs = predictor(img)  # "instances" with fields: pred_boxes, pred_masks, scores, etc.

        # Visualize
        v = Visualizer(img[:, :, ::-1], scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = out.get_image()[:, :, ::-1]  # convert back to BGR for OpenCV

        # Save
        save_path = os.path.join(args.output_folder, img_name)
        cv2.imwrite(save_path, result)
        print(f"Saved inference overlay: {save_path}")

if __name__ == "__main__":
    main()