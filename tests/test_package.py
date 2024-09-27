from plant_growth_tracker import process_images

# Path to a folder containing test images
image_folder = 'data'

# Process images using the package
df = process_images(
    image_folder_path=image_folder,
    output_csv='test_results.csv',
    segmentation_type='total_plant_area'  # or 'individual_leaf_area'
)

print(df)
