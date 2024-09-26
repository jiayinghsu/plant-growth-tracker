# Plant Growth Tracker

An API for segmenting plant leaf images and tracking plant growth over time.

## Overview

The **Plant Growth Tracker** is a FastAPI application that allows users to upload images and videos of plants to perform total plant area segmentation and individual leaf area segmentation. The results are returned in a format compatible with pandas DataFrames for easy data analysis.

## Features

- **Total Plant Area Segmentation**: Calculate the total area of each plant in an image or video.
- **Individual Leaf Area Segmentation**: Calculate the area of each leaf for each plant.
- **Batch Processing**: Support for processing multiple images at once.
- **Video Processing**: Extract frames from videos and perform segmentation.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/plant-growth-tracker.git
   cd plant-growth-tracker
