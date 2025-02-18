import os
import random
import argparse
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode

def register_dataset_if_needed(dataset_name, ann_file, image_root, thing_classes):
    if dataset_name in DatasetCatalog.list():
        old_things = MetadataCatalog.get(dataset_name).thing_classes
        if old_things == thing_classes:
            print(f"'{dataset_name}' already registered with {old_things}. Skipping.")
            return
        else:
            print(f"Removing old registration for '{dataset_name}' (old classes={old_things}).")
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)
    # Now register fresh
    register_coco_instances(dataset_name, {"thing_classes": thing_classes}, ann_file, image_root)

def visualize_sample(dataset_name):
    """
    Visualize one random sample from the dataset in an OpenCV window.
    (Skip this if in a headless environment.)
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    sample = random.choice(dataset_dicts)
    img_path = sample["file_name"]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not open image: {img_path}")
        return

    vis = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW
    )
    out = vis.draw_dataset_dict(sample)
    cv2.imshow("Sample Visualization", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Data preparation and registration.")
    parser.add_argument("--dataset-name", default="leaf_dataset", help="Name to register the dataset under.")
    parser.add_argument("--images", default="data/images", help="Folder path to images.")
    parser.add_argument("--ann", default="data/_annotations.coco.json", help="Path to COCO JSON file.")
    parser.add_argument("--task", default="leaf", choices=["plant", "leaf"],
                        help="Which single class to expect: 'plant' or 'leaf'.")
    parser.add_argument("--visualize", action="store_true", help="Visualize a random sample after registration.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Decide the single class label
    thing_classes = [args.task]  # e.g. ["plant"] or ["leaf"]

    # Register if needed
    register_dataset_if_needed(
        dataset_name=args.dataset_name,
        ann_file=args.ann,
        image_root=args.images,
        thing_classes=thing_classes
    )

    # Optionally visualize
    if args.visualize:
        visualize_sample(args.dataset_name)

if __name__ == "__main__":
    main()