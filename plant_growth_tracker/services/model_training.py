import os
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def read_batch(data, visualize_data=False):
    """
    Randomly select one training sample, load its image and mask,
    resize them so that the larger dimension is 1024, and generate a binary mask
    and one point per instance. The mask is assumed to have background (0) and
    instance labels (>=1).
    """
    # Randomly select one sample.
    ent = data[np.random.randint(len(data))]
    
    # Read the image.
    img = cv2.imread(ent["image"])
    if img is None:
        print(f"Error: Could not read image from {ent['image']}")
        return None, None, None, 0
    img = img[..., ::-1]  # Convert from BGR to RGB

    # Read the mask (instance segmentation mask with unique labels for each leaf).
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    if ann_map is None:
        print(f"Error: Could not read mask from {ent['annotation']}")
        return None, None, None, 0

    # Resize image and mask if necessary (scale so that the larger dimension is 1024).
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # Create a combined binary mask for all instances.
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []
    # The unique values in ann_map represent background (0) and each instance (>=1).
    instance_ids = np.unique(ann_map)[1:]  # Skip background.
    for inst in instance_ids:
        mask_inst = (ann_map == inst).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask_inst)
    
    # Erode the binary mask to avoid boundary artifacts.
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
        # Sample one prompt point per instance.
        for _ in instance_ids:
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])
    points = np.array(points)
    
    if visualize_data:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')
    
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
    
        plt.subplot(1, 3, 3)
        plt.title('Mask with Sampled Points')
        plt.imshow(binary_mask, cmap='gray')
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Adjust shapes: binary_mask becomes (C, H, W) and points are expanded for SAM2.
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # (H, W, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))         # (C, H, W)
    points = np.expand_dims(points, axis=1)                # shape becomes (N, 1, 2)
    
    return img, binary_mask, points, len(instance_ids)

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the SAM2 model for leaf instance segmentation using a custom dataset."
    )
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to train_data.json (output from data preparation).")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                        help="Path to the SAM2 checkpoint file (e.g., sam2_hiera_small.pt).")
    parser.add_argument("--model_cfg", type=str, required=True,
                        help="Path to the SAM2 model config YAML (e.g., sam2_hiera_s.yaml).")
    parser.add_argument("--no_of_steps", type=int, default=3000,
                        help="Number of training steps.")
    parser.add_argument("--fine_tuned_model_name", type=str, default="fine_tuned_sam2",
                        help="Prefix for the fine-tuned model checkpoint filenames.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (e.g., cuda or cpu).")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the training batch (optional).")
    args = parser.parse_args()

    # Load training data from JSON.
    with open(args.train_data, "r") as f:
        train_data = json.load(f)

    # Build the SAM2 model and wrap it in the predictor.
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)
    # Set the SAM2 components that will be trained.
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Setup optimizer, scaler for mixed precision, and learning rate scheduler.
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

    mean_iou = 0.0
    for step in range(1, args.no_of_steps + 1):
        with torch.cuda.amp.autocast():
            image, mask, input_point, num_instances = read_batch(train_data, visualize_data=args.visualize)
            if image is None or mask is None or num_instances == 0:
                continue

            # Create a label tensor for the prompt points.
            input_label = np.ones((num_instances, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue
            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            # Prepare prompt inputs for SAM2.
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            # Get sparse and dense embeddings from the prompt encoder.
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            # Extract high resolution features.
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            # Run the mask decoder.
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            # Prepare ground truth and predicted masks.
            gt_mask = torch.tensor(mask.astype(np.float32)).to(args.device)
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) -
                        (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter + 1e-6)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            # Apply gradient accumulation.
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            scheduler.step()

            if step % 500 == 0:
                model_filename = f"{args.fine_tuned_model_name}_{step}.torch"
                torch.save(predictor.model.state_dict(), model_filename)
                print(f"Saved model checkpoint to {model_filename}")

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            if step % 100 == 0:
                print(f"Step {step}: Accuracy (IoU) = {mean_iou:.4f}")

if __name__ == "__main__":
    main()