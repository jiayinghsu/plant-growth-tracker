import os
import shutil
import pandas as pd
import numpy as np
from plant_growth_tracker.services.image_processor import process_total_plant_area_images
from plant_growth_tracker.core.utils import load_image, default_preprocess_image
import cv2
import matplotlib.pyplot as plt
import logging

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_and_save(image: np.ndarray, preprocessed: np.ndarray, mask: np.ndarray, contours, save_path: str):
    """
    Visualizes and saves the preprocessing steps for debugging.
    
    Args:
        image (np.ndarray): Original image.
        preprocessed (np.ndarray): Preprocessed image.
        mask (np.ndarray): Binary mask.
        contours: Detected contours.
        save_path (str): Path to save the visualization.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(preprocessed)
    axs[0, 1].set_title('Preprocessed Image')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(mask, cmap='gray')
    axs[1, 0].set_title('Binary Mask')
    axs[1, 0].axis('off')
    
    image_contours = preprocessed.copy()
    cv2.drawContours(image_contours, contours, -1, (255, 0, 0), 2)
    axs[1, 1].imshow(image_contours)
    axs[1, 1].set_title('Detected Contours')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def custom_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Custom preprocessing steps:
    1. Resize to 2000x2000 pixels.
    2. Crop 8% from each border (top, bottom, left, right).
    3. Invoke default preprocessing.
    
    Args:
        image (np.ndarray): The input image.
    
    Returns:
        np.ndarray: The preprocessed image.
    """
    # Step 1: Resize to 2000x2000 pixels
    desired_size = (2000, 2000)
    resized_image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)
    
    # Step 2: Crop 8% from each border
    height, width = resized_image.shape[:2]
    crop_percent = 0.08
    crop_pixels_h = int(height * crop_percent)
    crop_pixels_w = int(width * crop_percent)
    
    cropped_image = resized_image[
        crop_pixels_h : height - crop_pixels_h,
        crop_pixels_w : width - crop_pixels_w
    ]
    
    # Step 3: Apply default preprocessing
    preprocessed_image = default_preprocess_image(cropped_image, visualize=False)
    
    return preprocessed_image

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Paths
    image_folder_path = 'data'
    output_dir = 'tests/output'
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    results_csv_path = os.path.join(output_dir, 'plant_area_results.csv')
    
    # Ensure output directories are clean and exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    ensure_dir(visualizations_dir)
    
    # Collect all image paths
    image_paths = [
        os.path.join(image_folder_path, img)
        for img in os.listdir(image_folder_path)
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))
    ]
    
    logging.info(f"Found {len(image_paths)} images to process.")
    
    # Process images with custom preprocessing
    results = process_total_plant_area_images(
        image_paths=image_paths,
        preprocessing_function=custom_preprocess  # Pass the custom preprocessing function
    )
    
    logging.info(f"Processed {len(results)} plant detections across all images.")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Step 1: Select Top 6 Largest Plants per Image
    df_sorted = df.sort_values(['image_name', 'plant_area'], ascending=[True, False])
    df_top6 = df_sorted.groupby('image_name').head(6).copy()
    
    # Step 2: Re-Rank plant_id from 1 to 6 for Each Image
    df_top6['plant_id'] = df_top6.groupby('image_name').cumcount() + 1
    logging.info("Re-ranked plant_id from 1 to 6 for each image.")
    
    # Step 3: Calculate average_plant_area per image by summing plant_area and dividing by 6
    sum_plant_area = df_top6.groupby('image_name')['plant_area'].sum().reset_index()
    sum_plant_area.rename(columns={'plant_area': 'sum_plant_area'}, inplace=True)
    
    # Merge sum_plant_area back to the top 6 dataframe
    df_final = pd.merge(df_top6, sum_plant_area, on='image_name', how='left')
    
    # Add average_plant_area by dividing sum_plant_area by 6
    df_final['average_plant_area'] = df_final['sum_plant_area'] / 6
    
    # Drop the sum_plant_area column as it's no longer needed
    df_final.drop(columns=['sum_plant_area'], inplace=True)
    
    # Step 4: Verify that each image has exactly 6 plants after selection
    plant_counts = df_final['image_name'].value_counts()
    images_with_incorrect_plant_counts = plant_counts[plant_counts != 6]
    
    if not images_with_incorrect_plant_counts.empty:
        logging.warning("Some images do not have exactly 6 plants after selecting top 6:")
        for image, count in images_with_incorrect_plant_counts.items():
            logging.warning(f"{image}: {count} plants")
    else:
        logging.info("All images have exactly 6 plants after selecting top 6.")
    
    # Step 5: Calculate average_plant_area statistics
    avg_area_stats = df_final['average_plant_area'].describe()
    logging.info(f"Average Plant Area Statistics:\n{avg_area_stats}")
    
    # Step 6: Save the final DataFrame
    ensure_dir(os.path.dirname(results_csv_path))
    df_final.to_csv(results_csv_path, index=False)
    logging.info(f"Final results saved to {results_csv_path}")
    
    # **Step 7: Generate and save visualizations**
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            image = load_image(image_path)
        except ValueError as e:
            logging.error(f"Error loading image {image_name}: {e}")
            continue
        
        # Apply custom preprocessing
        preprocessed_image = custom_preprocess(image)
        
        # Convert to grayscale and binary mask
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization
        visualization_path = os.path.join(visualizations_dir, f"visualization_{image_name}.png")
        visualize_and_save(image, preprocessed_image, binary, contours, visualization_path)
    
    print(f"\nProcessing complete. Results saved to {results_csv_path}")
    print(f"Visualizations saved to {visualizations_dir}")

if __name__ == "__main__":
    main()