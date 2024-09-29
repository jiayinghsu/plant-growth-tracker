# test_package.py

from plant_growth_tracker import process_images

# Replace with the actual path to your image folder
image_folder_path = 'data'

# Process images
df = process_images(
    image_folder_path=image_folder_path,
    segmentation_type='total_plant_area'  # or 'individual_leaf_area'
)

# Print the results
print(df)
