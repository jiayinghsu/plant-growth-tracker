import os
import json
import argparse
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def polygon_to_mask(polygons, height, width):
    """Given a list of polygons (each a list of [x1, y1, x2, y2, …]), create a mask image.
       Each polygon will be filled with a unique integer (starting at 1).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for label, poly in enumerate(polygons, start=1):
        # Convert the list of points into a Nx2 array of int32
        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=label)
    return mask

def main():
    parser = argparse.ArgumentParser(
        description="Prepare train and test data for leaf instance segmentation using SAM2."
    )
    parser.add_argument("--ann_json", type=str, required=True,
                        help="Path to the annotation JSON file (COCO format).")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to the folder containing images.")
    parser.add_argument("--masks_dir", type=str, required=True,
                        help="Path to output folder to save generated mask images.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save train_data.json and test_data.json.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of images to use for testing.")
    args = parser.parse_args()

    os.makedirs(args.masks_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the annotation JSON.
    with open(args.ann_json, "r") as f:
        data = json.load(f)

    # Build a mapping from image_id to its annotations (each with its segmentation polygon).
    imgid_to_polys = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg[0], list):
            poly = []
            for sub in seg:
                poly.extend(sub)
            seg = [poly]
        # Save each segmentation polygon for this image.
        imgid_to_polys.setdefault(img_id, []).extend(seg)

    # Create a dictionary for quick lookup of image info by id.
    images_info = {img["id"]: img for img in data.get("images", [])}

    samples = []
    # Process each image.
    for img_id, img_info in images_info.items():
        file_name = img_info["file_name"]
        img_path = os.path.join(args.images_dir, file_name)
        # Use the provided image dimensions.
        height = img_info.get("height")
        width = img_info.get("width")
        # Get all polygons for this image (if any).
        polygons = imgid_to_polys.get(img_id, [])
        if not polygons:
            # Skip images without annotations.
            continue
        # Create the mask.
        mask = polygon_to_mask(polygons, height, width)
        # Save the mask image.
        mask_filename = os.path.splitext(file_name)[0] + "_mask.png"
        mask_path = os.path.join(args.masks_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        samples.append({
            "image": os.path.abspath(img_path),
            "annotation": os.path.abspath(mask_path)
        })

    if not samples:
        print("No samples with annotations were found.")
        return

    # Split samples into training and testing sets.
    train_samples, test_samples = train_test_split(samples, test_size=args.test_size, random_state=42)

    train_out = os.path.join(args.output_dir, "train_data.json")
    test_out = os.path.join(args.output_dir, "test_data.json")
    with open(train_out, "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(test_out, "w") as f:
        json.dump(test_samples, f, indent=2)

    print(f"Prepared {len(train_samples)} training and {len(test_samples)} testing samples.")
    print(f"Train data saved to {train_out}")
    print(f"Test data saved to {test_out}")

if __name__ == "__main__":
    main()
