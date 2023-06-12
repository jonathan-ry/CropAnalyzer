Crop Analysis Software for Yield Prediction and NDVI Processing

Built using Python

Introduction:
The Crop Analysis Software is a comprehensive tool designed to analyze crop data obtained from drone imagery. The software processes a series of images captured by drones, stitches them together, calculates the Normalized Difference Vegetation Index (NDVI), and leverages data provided by farmers to predict crop yield. By integrating advanced image processing algorithms and data analysis techniques, this software aims to assist farmers in making informed decisions regarding crop management, resource allocation, and yield optimization.

Key Features:

Image Stitching:
The software ingests a series of images captured by drones flying over farmland. It employs advanced image stitching algorithms to seamlessly combine the individual images, creating a single, high-resolution composite image of the entire field. This stitched image serves as the basis for subsequent analysis.

NDVI Calculation:
Normalized Difference Vegetation Index (NDVI) is a widely used metric for assessing crop health and vigor. The software utilizes the stitched image and applies spectral analysis techniques to calculate NDVI values for each pixel in the image. NDVI values range from -1 to 1, with higher values indicating healthier and more abundant vegetation.

Crop Health Visualization:
The software generates visual representations of the NDVI values across the crop field, providing farmers with a clear and intuitive understanding of the health and vigor of their crops. These visualizations can highlight areas of concern, such as potential disease outbreaks, nutrient deficiencies, or water stress, allowing farmers to take targeted remedial actions.

Yield Prediction:
In addition to NDVI analysis, the software integrates data provided by farmers, such as historical crop yields, planting dates, weather conditions, and farming practices. By leveraging machine learning and statistical modeling techniques, the software predicts crop yield based on the available data. These predictions help farmers in estimating future harvests, optimizing resource allocation, and making informed business decisions.



Imports and Libraries Requirements
PyQt5
numpy
pillow==4.1.0
xmltodict

matplotlib
pandas
sklearn

pip install -i http://pypi.douban.com/simple --trusted-host pypi.douban.com opencv-contrib-python==3.4.2.17


Python36
include-system-site-packages = false
version = 3.6.5


Code snippet for image stitching sourced from https://github.com/alexhagiopol/orthomosaic

