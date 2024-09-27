## Customizing Preprocessing

You can customize the preprocessing steps by providing your own preprocessing function. This function should accept an image (as a NumPy array) and return the preprocessed image.

### Example 1: Using Default Preprocessing
```python
from plant_growth_tracker import process_images

df = process_images(
    image_folder_path='path/to/images',
    segmentation_type='total_plant_area',
)
print(df)
```

### Example 2: Skipping Preprocessing
```python
from plant_growth_tracker import process_images

def no_preprocessing(image):
    return image  # Return the image as-is

df = process_images(
    image_folder_path='path/to/images',
    segmentation_type='total_plant_area',
    preprocessing_function=no_preprocessing,
)
print(df)
```

### Example 3: Using a Custom Preprocessing Function

```python
from plant_growth_tracker import process_images
import cv2

def custom_preprocessing(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

df = process_images(
    image_folder_path='path/to/your/image_folder',
    segmentation_type='total_plant_area',
    preprocessing_function=custom_preprocessing,
)
```
