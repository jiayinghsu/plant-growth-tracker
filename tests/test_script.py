import sys
import os

# Calculate the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

# Now you can import your package
from plant_growth_tracker import process_images

# Provide the path to your image folder
image_folder_path = 'data'  # Replace with your actual path

# Process images
df = process_images(
    image_folder_path=image_folder_path,
    segmentation_type='total_plant_area'  # or 'individual_leaf_area'
)

# Print the results
print(df)
