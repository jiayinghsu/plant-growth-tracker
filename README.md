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
  - [Training a Custom Model](#training-a-custom-model)
    - [Preparing Your Dataset](#preparing-your-dataset)
    - [Training Procedure](#training-procedure)
  - [Using the Custom Trained Model](#using-the-custom-trained-model)
- [Additional Notes](#additional-notes)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Total Plant Area Calculation**: Process images or videos to calculate the total area occupied by plants.
- **Individual Leaf Area Calculation**: Segment individual leaves and calculate their areas using a custom-trained model.
- **Custom Model Training**: Train a custom segmentation model using your own dataset for improved accuracy.

---

## Installation

### Prerequisites

- **Python**: Version 3.6 or higher.
- **Hardware**: A GPU is recommended for training the custom model.
- **Dependencies**: Install the required packages using:

  ```bash
  pip install -r requirements.txt
  ```

  **`requirements.txt`**:

  ```text
  numpy
  pandas
  opencv-python
  pydantic
  pillow
  pillow-heif
  scikit-image
  torch
  torchvision
  transformers
  tqdm
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
   git clone https://github.com/yourusername/plant-growth-tracker.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd plant-growth-tracker
   ```

3. **Install the Package**:

   ```bash
   pip install .
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

# Save or print the results
print(df)
df.to_csv('total_plant_area_results.csv', index=False)
```

#### Individual Leaf Area

To calculate individual leaf areas, you need to train a custom model first (see [Training a Custom Model](#training-a-custom-model)). Once you have a trained model, use it as follows:

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

# Save or print the results
print(df)
df.to_csv('individual_leaf_area_results.csv', index=False)
```

### Training a Custom Model

#### Preparing Your Dataset

Organize your images and corresponding masks in separate directories:

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

- **Images**: Place all your training images in the `images` directory.
- **Masks**: Create binary masks for each image where the leaves are white (pixel value 255) and the background is black (pixel value 0).
- **Naming Convention**: Ensure that each mask filename corresponds exactly to its image filename (e.g., `image1.jpg` and `image1.png`).

#### Training Procedure

1. **Create a Training Script**

   Create a Python script named `train_model.py`:

   ```python
   from plant_growth_tracker.services.model_training import train_custom_model

   images_dir = 'dataset/images'
   masks_dir = 'dataset/masks'
   model_save_path = 'trained_model'
   processor_save_path = 'trained_processor'

   train_custom_model(
       images_dir=images_dir,
       masks_dir=masks_dir,
       model_save_path=model_save_path,
       processor_save_path=processor_save_path,
       num_epochs=10,
       batch_size=2,
       learning_rate=1e-5,
       device='cuda' if torch.cuda.is_available() else 'cpu'
   )
   ```

2. **Run the Training Script**

   Execute the script:

   ```bash
   python train_model.py
   ```

   - **Parameters**:
     - `num_epochs`: Number of times the entire dataset is passed through the model.
     - `batch_size`: Number of samples processed before the model is updated.
     - `learning_rate`: Step size at each iteration while moving toward a minimum of a loss function.
     - `device`: Specify `'cuda'` for GPU acceleration or `'cpu'` for CPU.

3. **Monitor Training**

   - The script will output the training loss after each epoch.
   - Adjust parameters based on the training performance and available computational resources.

### Using the Custom Trained Model

After training, use your custom model to process new images:

```python
from plant_growth_tracker import process_images

image_folder_path = 'path/to/new/images'
custom_model_paths = {
    'model_path': 'trained_model',
    'processor_path': 'trained_processor'
}

df = process_images(
    image_folder_path=image_folder_path,
    segmentation_type='individual_leaf_area',
    custom_model_paths=custom_model_paths
)

# Save or print the results
print(df)
df.to_csv('individual_leaf_area_results.csv', index=False)
```

- **`custom_model_paths`**: Dictionary containing the paths to your trained model and processor.
- **Output**: The `process_images` function returns a pandas DataFrame with the results.

---

## Additional Notes

- **GPU Recommendation**: For training and inference on large datasets, a GPU is highly recommended to speed up processing.
- **Data Quality**: The accuracy of the segmentation depends on the quality of your training data. Ensure masks are accurately annotated.
- **Parameter Adjustment**: You may need to adjust thresholds and parameters in the code to suit your specific dataset.
- **Dependencies**: Ensure all required packages are installed. Use:

  ```bash
  pip install -r requirements.txt
  ```

- **Error Handling**: The package includes basic error handling. If you encounter issues, please check that all file paths and parameters are correct.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Click on the 'Fork' button at the top right of the repository page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/yourusername/plant-growth-tracker.git
   ```

3. **Create a New Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**: Implement your feature or fix.

5. **Commit and Push**:

   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request**: Go to the original repository and click on 'New Pull Request'.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.