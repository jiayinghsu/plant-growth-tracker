from plantcv import plantcv as pcv
import cv2

class options:
    def __init__(self):
        self.image = "/Users/jiayingxu/ucsd/plant_growth_tracker/samples/col_1_25.png"
        self.debug = "plot"
        self.writeimg = False
        self.outdir = "/Users/jiayingxu/ucsd/plant_growth_tracker/results"

# Get options
args = options()

# Set debug to the global parameter
pcv.params.debug = args.debug

# Read image
# Inputs:
#   filename - Image file to be read in
#   mode - Return mode of image; either "native" (default), "rgb", "gray", or "csv"

img, path, filename = pcv.readimage(filename=args.image)

# Crop the image down to focus on just one plant
crop_img = img[100:550,1000:1750]
pcv.plot_image(crop_img)

gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

# remove the excessive noise using THRESHOLD
thresh, img_binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
pcv.plot_image(img_binary)

# Find objects
# Inputs:
#   img = image that the objects will be overlayed
#   mask = what is used for object detection
id_objects, obj_hierarchy = pcv.find_objects(img=crop_img, mask=img_binary)

# Combine objects
# Inputs:
#   img = RGB or grayscale image data for plotting
#   contours = contour list
#   hierarchy = contour hierarchy array
obj, mask =  pcv.object_composition(img=crop_img, contours=id_objects, hierarchy=obj_hierarchy)

# Apply mask
# Inputs:
# img = RGB or grayscale image data
# mask = binary mask image data
# mask_color = "white" or "black"
masked = pcv.apply_mask(img=crop_img, mask=mask, mask_color="black")

# Use watershed segmentation

# Inputs:
#   rgb_img = RGB image data
#   mask = binary image, single channel, object in white and background in black
#   distance = minimum distance of local maximum, lower values are more sensitive,
#               and segments more objects (default: 10)
#   label = optional label parameter, modifies the variable name of observations recorded. (default `label="default"`)filled_img = pcv.morphology.fill_segments(mask=cropped_mask, objects=edge_objects)
analysis_images = pcv.watershed_segmentation(rgb_img=crop_img, mask=mask, distance=15, label="default")

# The save results function will take the measurements stored when running any PlantCV analysis functions, format,
# and print an output text file for data analysis. The Outputs class stores data whenever any of the following functions
# are ran: analyze_bound_horizontal, analyze_bound_vertical, analyze_color, analyze_nir_intensity, analyze_object,
# fluor_fvfm, report_size_marker_area, watershed. If no functions have been run, it will print an empty text file
pcv.outputs.save_results(filename='segmentation_tutorial_results.json')

import pandas as pd
pdObj = pd.read_json('segmentation_tutorial_results.json', orient="index")
print(pdObj)