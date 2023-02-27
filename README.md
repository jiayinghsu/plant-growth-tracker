# Tracking plant growth using computer vision
Please use, cite, and contribute to Plant growth tracker (PGT)! If you have questions, please submit them via the [GitHub issue page](https://github.com/jiayinghsu/plant_growth_tracker/issues). 
***

## Introduction 
PGT is an open-source pipeline for tracking plant's growth. PGT handles the problems of inconsistent coordinates, angles, lighting and distances during the initial phase of data collection. PGT allows more reliable comparison among different experimental conditions by minimizing these technical caveats. 


## Quick Links

 - [0-Batch convert HEIC to PNG](https://github.com/jiayinghsu/plant_growth_tracker/blob/main/0-heic_to_png.py)
 - [1-Segment plants, plates and background in images](https://github.com/jiayinghsu/plant_growth_tracker/blob/main/1-segment_images.py)
 - [2-Remove backgrounds and standardize coordinates](https://github.com/jiayinghsu/plant_growth_tracker/blob/main/2-remove_bacgrounds_align_coordinates.py) 
 - [3-Analyze plant size and output csv files](https://github.com/jiayinghsu/plant_growth_tracker/blob/main/3-calculate_size.py)  
 - [4-Leaf segmentation using Mask-RCNN](https://github.com/jiayinghsu/plant_growth_tracker/blob/main/4-leaf_segmentation.py)  
 

 ## Code References

 - https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
 - https://stackoverflow.com/questions/57283802/remove-small-whits-dots-from-binary-image-using-opencv-python
 - https://stackoverflow.com/questions/60780831/python-how-to-cut-out-an-area-with-specific-color-from-image-opencv-numpy


## Work in Progress

 - [ ] Segment 3D point clouds of plant organs such as leaves and stems to automate plant growth monitoring and quantify the strss levels the plant is experiencing.
 - [ ] Experiment with algorithms such as PointNet++, ASIS, SGPN, and PlantNet.


## Issues 

Please file any PGT suggestions/issues/bugs via our 
[GitHub issues page](https://github.com/jiayinghsu/plant_growth_tracker/issues). Please check to see if any related 
issues have already been filed.

***
