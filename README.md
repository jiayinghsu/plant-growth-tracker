# Plant Growth Tracker

## Overview

The **Plant Growth Tracker** is a Python package designed to help researchers and agronomists analyze plant growth by processing images and videos to calculate total plant areas and individual leaf areas. It leverages advanced image processing techniques and allows for custom training of segmentation models tailored to your dataset.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Processing Images](#processing-images)
    - [Total Plant Area](#total-plant-area)
    - [Individual Leaf Area](#individual-leaf-area)
  - [Custom Preprocessing](#custom-preprocessing)
  - [Training a Custom Model](#training-a-custom-model)
    - [Preparing Your Dataset](#preparing-your-dataset)
    - [Data Preparation for Training](#data-preparation-for-training)
    - [Training Procedure](#training-procedure)
  - [Running Inference with the Custom Trained Model](#running-inference-with-the-custom-trained-model)
    - [Overlay Inference Use Case](#overlay-inference-use-case)
    - [CSV Inference Use Case](#csv-inference-use-case)
  - [Running Custom Analysis](#running-custom-analysis)
- [Additional Notes](#additional-notes)
- [Publication](#publication)
- [License](#license)

---

## Features

- **Total Plant Area Calculation**: Process images or videos to calculate the total area occupied by plants.
- **Individual Leaf Area Calculation**: Segment individual leaves and calculate their areas using a custom-trained model.
- **Custom Model Training**: Train a custom segmentation model using your own dataset for improved accuracy.
- **Custom Preprocessing**: Apply custom preprocessing steps to enhance image quality before analysis.

---

## Installation

### Prerequisites

- **Python**: Version 3.6 or higher.
- **Hardware**: A GPU is recommended for training the custom model.
- **Dependencies**: Install the required packages using:

  ```bash
  pip install -r requirements.txt
  ```

  **Example `requirements.txt`:**

  ```text
  fastapi
  uvicorn[standard]
  opencv-python==4.7.0.72
  numpy==1.24.4
  pandas
  pydantic
  torch==2.0.1
  torchvision==0.15.2
  pillow
  pillow-heif
  scikit-image 
  transformers
  tqdm
  matplotlib
  git+https://github.com/facebookresearch/segment-anything.git
  ```

### Install the Package

#### From PyPI (Coming Soon)

```bash
pip install plant-growth-tracker
```

#### From Source

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/jiayinghsu/plant-growth-tracker.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd plant-growth-tracker
   ```

3. **Install the Package in Editable Mode**:

   ```bash
   pip install -e .
   ```

   > **Note:** Installing in editable mode (`-e`) allows you to make changes to the source code without reinstalling the package.

---

## Repository Structure

An example of the repository layout is:

```
plant-growth-tracker
├── LICENSE
├── MANIFEST.in
├── README.md
├── plant_growth_tracker
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── custom_model.py
│   │   └── schemas.py
│   └── services
│       ├── __init__.py
│       ├── data_prep.py           
│       ├── image_processor.py
│       ├── model_training.py      
│       ├── segmentation.py
│       └── video_processor.py
├── use_cases
│   ├── __init__.py              
│   ├── inference_folder.py      # Inference: overlay segmentation on images
│   └── inference_to_csv.py      # Inference: output CSV summary of segmentation results
├── pyproject.toml
├── requirements.txt
├── setup.py
└── tests
    ├── __init__.py
    ├── test_package.py
    └── test_script.py
```

---

## Usage

### Processing Images

#### Total Plant Area

To calculate the total plant area in images:

```python
from plant_growth_tracker import process_images

image_folder_path = 'path/to/your/image_folder'

df = process_images(
    image_folder_path=image_folder_path,
    segmentation_type='total_plant_area'
)

print(df)
df.to_csv('total_plant_area_results.csv', index=False)
```

#### Individual Leaf Area

To calculate individual leaf areas, you may need to first train a custom model (see below). Once you have a trained model, use it as follows:

```python
from plant_growth_tracker import process_images

image_folder_path = 'path/to/your/image_folder'
custom_model_paths = {
    'model_path': 'path/to/save/your_trained_model',
    'processor_path': 'path/to/save/your_trained_processor'
}

df = process_images(
    image_folder_path=image_folder_path,
    segmentation_type='individual_leaf_area',
    custom_model_paths=custom_model_paths
)

print(df)
df.to_csv('individual_leaf_area_results.csv', index=False)
```

### Custom Preprocessing

Apply custom preprocessing steps before analysis to improve segmentation accuracy:

```python
from plant_growth_tracker import process_images, custom_preprocess

image_folder_path = 'path/to/your/image_folder'

df = process_images(
    image_folder_path=image_folder_path,
    segmentation_type='total_plant_area',
    preprocessing_function=custom_preprocess
)

print(df)
df.to_csv('total_plant_area_results_custom.csv', index=False)
```

---

## Training a Custom Model

### Preparing Your Dataset

Organize your dataset as follows:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

- **Images**: Place all training images in the `images` directory.
- **Masks**: Create binary masks for each image (leaves in white, background in black).
- **Naming Convention**: Each mask file should correspond to its image (e.g., `image1.jpg` and `image1.png`).

### Data Preparation for Training

Run the data preparation script to convert COCO-style annotations into masks and generate JSON files:

```bash
python plant_growth_tracker/services/data_prep.py \
    --ann_json /path/to/annotations.json \
    --images_dir /path/to/images_folder \
    --masks_dir /path/to/output_masks \
    --output_dir /path/to/output_json \
    --test_size 0.2
```

**Expected Output:**

- Mask images saved in the specified `masks_dir`.
- Two JSON files (`train_data.json` and `test_data.json`) in the specified `output_dir`.

### Training Procedure

1. **Create a Training Script**

   You can directly run the training script located at `plant_growth_tracker/services/model_training.py`. For example, from your terminal run:

   ```bash
   python plant_growth_tracker/services/model_training.py \
       --train_data /path/to/output_json/train_data.json \
       --sam2_checkpoint /path/to/sam2_hiera_small.pt \
       --model_cfg /path/to/sam2_hiera_s.yaml \
       --no_of_steps 3000 \
       --fine_tuned_model_name fine_tuned_sam2 \
       --lr 0.0001 \
       --accumulation_steps 4 \
       --device cuda
   ```

   **Parameters:**

   - `--train_data`: Path to your `train_data.json`
   - `--sam2_checkpoint`: SAM2 checkpoint file path
   - `--model_cfg`: SAM2 configuration file path
   - `--no_of_steps`: Number of training steps
   - `--fine_tuned_model_name`: Prefix for saved checkpoints
   - `--lr`: Learning rate
   - `--accumulation_steps`: Gradient accumulation steps
   - `--device`: `'cuda'` for GPU or `'cpu'`

2. **Monitor Training**

   - The script prints training loss and saves checkpoints periodically.
   - Adjust parameters based on your dataset and available hardware.

---

## Running Inference with the Custom Trained Model

After training, use your fine-tuned model to perform inference on new images. Two use cases are provided in the **use_cases** folder.

### Overlay Inference Use Case

This script overlays predicted segmentation masks (with instance IDs) on input images and saves the results.

Run the script with:

```bash
python use_cases/inference_folder.py \
  --images_dir /path/to/input_images \
  --output_dir /path/to/output_overlays \
  --sam2_checkpoint /path/to/sam2_hiera_small.pt \
  --model_cfg /path/to/sam2_hiera_s.yaml \
  --fine_tuned_model_weights /path/to/fine_tuned_sam2_1000.torch \
  --num_points 30 --device cuda
```

**Expected Output:**

- Processed images with overlaid segmentation masks are saved in the specified output directory.

### CSV Inference Use Case

This script processes images and outputs a CSV file summarizing each detected leaf's properties (instance ID, image origin, class, area, centroid).

Run the script with:

```bash
python use_cases/inference_to_csv.py \
  --images_dir /path/to/input_images \
  --output_csv /path/to/results.csv \
  --sam2_checkpoint /path/to/sam2_hiera_small.pt \
  --model_cfg /path/to/sam2_hiera_s.yaml \
  --fine_tuned_model_weights /path/to/fine_tuned_sam2_1000.torch \
  --num_points 30 --device cuda
```

**Expected Output:**

- A CSV file (e.g., `results.csv`) listing for each image:
  - **Object ID**: Instance number (restarts at 1 for each image)
  - **Image origin**: Filename of the image
  - **Class**: Fixed as `"leaf"`
  - **Area**: Number of pixels in the detected instance
  - **X Position** and **Y Position**: Centroid coordinates

---

## Running Custom Analysis

To run a custom analysis using the package, execute the test script:

```bash
python tests/test_package.py
```

- **Description:**  
  This script applies custom preprocessing and analysis, generating visualizations and CSV outputs in the `tests/output/` directory.

---

## Additional Notes

- **GPU Recommendation:**  
  A GPU is highly recommended for training and inference on large datasets.
- **Data Quality:**  
  The accuracy of segmentation depends on high-quality, accurately annotated masks.
- **Parameter Adjustment:**  
  You may need to adjust thresholds and hyperparameters to suit your dataset.
- **Dependencies:**  
  Ensure all required packages are installed via:

  ```bash
  pip install -r requirements.txt
  ```

- **Error Handling:**  
  Basic error handling is implemented. Verify all file paths and parameter values if issues arise.

---

## Publication

This tool package has been utilized in the following publication:

- **Arabidopsis transcriptome responses to low water potential using high-throughput plate assays**

  *[Link to the paper](https://elifesciences.org/articles/84747)*

  > **Citation:**
  >
  > Gonzalez, S., Swift, J., Yaaran, A., Xu, J., Miller, C., Illouz-Eliaz, N., Nery, J. R., Busch, W., Zait, Y., & Ecker, J. R. (2023). *Arabidopsis transcriptome responses to low water potential using high-throughput plate assays*. eLife, 12, e84747. [https://doi.org/10.7554/eLife.84747](https://doi.org/10.7554/eLife.84747)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.