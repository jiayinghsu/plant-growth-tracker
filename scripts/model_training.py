import os
import argparse
from datetime import datetime

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

# If data_preparation.py is in the same folder, import from it:
from data_preparation import register_dataset_if_needed

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Detectron2 model.")
    parser.add_argument("--dataset-name", default="leaf_dataset")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--ann", default="data/_annotations.coco.json")
    parser.add_argument("--task", default="leaf", choices=["plant", "leaf"])
    parser.add_argument("--output-dir", default="output_leaf")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    # e.g. thing_classes = ["plant"]
    thing_classes = [args.task]

    # 1) Make sure dataset is registered if needed
    register_dataset_if_needed(
        dataset_name=args.dataset_name,
        ann_file=args.ann,
        image_root=args.images,
        thing_classes=thing_classes
    )

    # 2) Setup the detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Using the same dataset for training
    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST  = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    # Single class
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # 3) Create an output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(args.output_dir, args.dataset_name, timestamp)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 4) Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"Training complete! Model final weights at: {os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')}")

if __name__ == "__main__":
    main()