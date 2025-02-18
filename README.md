# Plant Growth Tracker

This repository provides four Python scripts demonstrating how to train and run inference (both image overlay and CSV output) on a **single-class instance segmentation task** using [Detectron2](https://github.com/facebookresearch/detectron2). The **task** can be either **“plant”** or **“leaf”** segmentation, selectable at runtime via a simple `--task` argument.

---
## Table of Contents
1. [Installation](#installation)
2. [Scripts Overview](#scripts-overview)
3. [Data Preparation & Training](#data-preparation--training)
4. [Running Inference with Overlayed Output](#running-inference-with-overlayed-output)
5. [Running Inference with CSV Output](#running-inference-with-csv-output)
6. [Example CSV Format](#example-csv-format)

---

## Installation

1. **Create a Python 3.8+ environment** (e.g., with conda or virtualenv):
   ```bash
   conda create -n detectron2-env python=3.8
   conda activate detectron2-env
   ```

2. **Install PyTorch**  
   Refer to [PyTorch’s official site](https://pytorch.org/) to pick the correct command for your system (CPU or GPU). For example (CUDA 11.8):
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   ```
   Or CPU-only:
   ```bash
   pip install torch torchvision
   ```

3. **Install Detectron2**  
   The official instructions are [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). A quick command:
   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```
   (Alternatively, you can install a specific version/tag if needed.)

4. **Install Other Dependencies**  
   ```bash
   pip install opencv-python tqdm
   ```
   (Depending on your system, you might already have them.)

---

## Scripts Overview

We have four main scripts:

1. **`data_preparation.py`**  
   - Registers your custom dataset in COCO-format.  
   - Can optionally visualize one sample to confirm annotations.

2. **`model_training.py`**  
   - Trains an instance segmentation model (Mask R-CNN) on your custom dataset.  
   - Allows selecting a **single class** (`--task plant` or `--task leaf`).  

3. **`run_inference.py`**  
   - Runs inference on a folder of images.  
   - Saves overlayed images (with bounding boxes & masks) to an output folder.

4. **`run_inference_csv.py`**  
   - Runs inference on a folder of images.  
   - Saves a CSV file listing each object’s ID, image name, class, mask area (in pixels), and (x, y) centroid.

---

## Data Preparation & Training

Before training, you must have:

- A **folder of images** (`.jpg`, `.png`, etc.).  
- A **COCO-format annotation file** (e.g., `_annotations.coco.json`) that describes bounding boxes and segmentation polygons/bitmasks.  
  - We **do not** provide this data in the repository; you must supply your own images and annotation file.

### 1) Data Preparation

Use `data_preparation.py` to register your dataset. For example:

```bash
python data_preparation.py \
  --dataset-name my_dataset \
  --images /path/to/your/images \
  --ann /path/to/your/annotations.json \
  --task plant \
  --visualize
```

- **`--task`** can be `plant` or `leaf`. This sets your single class name for Detectron2’s metadata.  
- **`--visualize`** opens a random image to confirm your annotations. If you’re on a headless server, omit this flag.

### 2) Training

Use `model_training.py` to train a Mask R-CNN model:

```bash
python model_training.py \
  --dataset-name my_dataset \
  --images /path/to/your/images \
  --ann /path/to/your/annotations.json \
  --task plant \
  --output-dir output_plant \
  --max-iter 1000 \
  --lr 0.00025 \
  --batch-size 2
```

Parameters:
- **`--task plant`** or `--task leaf`: single class name.  
- **`--dataset-name my_dataset`**: must match the name you used for data registration (if already registered).  
- **`--images`**, **`--ann`**: your dataset’s paths.  
- **`--output-dir`**: where logs & final model weights are stored.  
- **`--max-iter`**: number of training iterations (increase for a larger dataset).  
- **`--lr`**: learning rate.  
- **`--batch-size`**: images per batch (2 is typical if GPU memory is limited).  

Once training is done, you’ll see something like:
```
Training complete! Final model weights: output_plant/my_dataset/2025-02-17-19h/model_final.pth
```

---

## Running Inference with Overlayed Output

After training, you have a `model_final.pth`. You can run inference on **any folder** of images using `run_inference.py`. This will save images with bounding boxes & masks drawn on them:

```bash
python run_inference.py \
  --task leaf \
  --weights-file /path/to/model_final.pth \
  --input-folder /path/to/test_images \
  --output-folder /path/to/output_overlay \
  --score-threshold 0.5
```

Where:
- `--task leaf` or `--task plant` to match the model’s single class.  
- `--weights-file` points to the `.pth` you trained.  
- `--input-folder` is the folder containing new images for inference.  
- `--output-folder` is where overlayed images will be saved.  
- `--score-threshold` is the confidence threshold (default 0.5).  

### Example Overlayed Result

Suppose you have an image named `sample_plant.jpg` in the input folder. After running the script, you might see an output image like:

<img src="https://via.placeholder.com/500x300.png?text=Example+Overlay" width="400" alt="Example overlay" />

*(Above is just a placeholder image; in your real usage, you’d see bounding boxes & masks for each detected plant/leaf.)*

---

## Running Inference with CSV Output

If you prefer a **CSV** report with instance details (object ID, class, area, centroid, etc.), use `run_inference_csv.py`:

```bash
python run_inference_csv.py \
  --task leaf \
  --weights-file /path/to/model_final.pth \
  --input-folder /path/to/test_images \
  --output-csv results/leaf_inference.csv \
  --score-threshold 0.5
```

No images are saved in this mode; instead, you get a CSV file with columns:
```
Object ID, Image origin, Class, Area, X Position, Y Position
```

---

## Example CSV Format

Below is a snippet of what the CSV might look like (for `--task leaf`), with each row representing one detected instance:

```
Object ID,Image origin,Class,Area,X Position,Y Position
1,"IMG_3618_Drought_wild type (allele 3 pot, pair 1).jpeg",leaf,6255.0,1036.1747402078338,1269.6495603517187
2,"IMG_3618_Drought_wild type (allele 3 pot, pair 1).jpeg",leaf,2888.0,896.8306786703602,1278.217105263158
3,"IMG_3618_Drought_wild type (allele 3 pot, pair 1).jpeg",leaf,8779.0,1104.5489235676046,987.2770247180772
4,"IMG_3618_Drought_wild type (allele 3 pot, pair 1).jpeg",leaf,2812.0,911.4068278805121,1157.6554054054054
5,"IMG_3618_Drought_wild type (allele 3 pot, pair 1).jpeg",leaf,5569.0,994.3943257317292,1030.3411743580534
...
```

- **Object ID**: an incremental number (1-based) per image.  
- **Image origin**: filename of the image.  
- **Class**: `"leaf"` (or `"plant"`, depending on your `--task`).  
- **Area**: mask area in pixels.  
- **X Position, Y Position**: centroid (pixel coordinates).

That’s it! You can now easily parse or analyze these CSV results in downstream workflows.