# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 03:38:28 2019

@author: cosmi

rawpy is an easy-to-use Python wrapper for the LibRaw library. 
rawpy works natively with numpy arrays and supports a lot of options, 
including direct access to the unprocessed Bayer data
It also contains some extra functionality for finding and repairing hot/dead pixels.
import rawpy.enhance for this

First, install the LibRaw library on your system.

pip install libraw.py

then install rawpy

pip install rawpy

"""

"""
Experimental Vegetation Index Mapping program using DJI Mavic 2 Pro 
JPEG 16-bit combo images taken using InfraBlue Filter 
Useful for Batch Processing Multiple Images

%(c)-J. Campbell MuonRay Enterprises 2019 
This Python script was created using the Spyder Editor

"""
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout, QLineEdit)
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QDir
from PyQt5.QtCore import QProcess
# import numpy as np  
from subprocess import call
from msg_box import Ui_MainWindow

import warnings
warnings.filterwarnings('ignore')
#import imageio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt  # For image viewing

from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

#dng reading requires libraw to work

import os
import rawpy


cols1 = ['blue', 'green', 'yellow', 'red']
cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
cols3 = ['gray', 'blue', 'green', 'yellow', 'red']
cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

def create_colormap(args):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols3)

#colour bar to match grayscale units
def create_colorbar(fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

with open('improcess_path.txt', 'r') as f:
    path = f.read()
# os.remove('size.txt')
print("This is the path", path)
for infile in os.listdir(path):
    print( "file : " + infile)
    if infile[-3:] == "JPG" or infile[-3:] == "png" :
    
       outfile = path + "/"+ infile[:-3] + "jpeg"
       rgb = plt.imread(path + "/"+infile)
       
       
       print( "new filename : " + outfile)
       # Extract Red, Green and Blue channels and save as separate files
       
       

       R = rgb[:,:,0]
       G = rgb[:,:,1]
       B = rgb[:,:,2]
       
              # Get the red band from the rgb image, and open it as a numpy matrix
#NIR = image[:, :, 0]
#ir = np.asarray(NIR, float)
              
       ir = (R).astype('float')
       
# Get one of the IR image bands (all bands should be same)
#blue = image[:, :, 2]

#r = np.asarray(blue, float)
       
       r = (B).astype('float')
       
       
       
       "For ENDVI"
#g = np.asarray(G, float)
       
       g = (G).astype('float')
       
       #(NIR + Green)
       irg = np.add(ir, g)
       
       "For SAVVI"

       
       L=0.5;
       
       rplusb = np.add(ir, r)
       rminusb = np.subtract(ir, r)
       oneplusL = np.add(1, L)
       
# Create a numpy matrix of zeros to hold the calculated NDVI values for each pixel
  # The NDVI image will be the same size as the input image

       
       ndvi = np.zeros(r.size)       
       
       ndvi = np.true_divide(np.subtract(irg, 2*r), np.add(irg, 2*r))
       
       #ndvi = np.true_divide(np.subtract(ir, r), np.add(ir, r))
       
       #savi = np.multiply(oneplusL , (np.true_divide((rminusb), np.add(rplusb, L))))
       #endvi = np.true_divide(np.subtract(irg, 2*r), np.add(irg, 2*r))

       
       
       
       fig, ax = plt.subplots()
       
       cols1 = ['blue', 'green', 'yellow', 'red']
       cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
       cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

       cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

       def create_colormap(args):
              return LinearSegmentedColormap.from_list(name='custom1', colors=cols3)


       #colour bar to match grayscale units
       def create_colorbar(fig, image):
              position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
              norm = colors.Normalize(vmin=-1., vmax=1.)
              cbar = plt.colorbar(image,
                                   cax=position,
                                   orientation='horizontal',
                                   norm=norm)
              cbar.ax.tick_params(labelsize=6)
              tick_locator = ticker.MaxNLocator(nbins=3)
              cbar.locator = tick_locator
              cbar.update_ticks()
              cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

       fig, ax = plt.subplots()
       image = ax.imshow(ndvi, cmap=create_colormap(colors))
       plt.axis('off')

       create_colorbar(fig, image)

       extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
       fig.savefig(outfile, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
       # plt.show()

window = QtWidgets.QMainWindow()
msg_box = Ui_MainWindow()
msg_box.setupUi(window)
window.show()

        # plt.show()
       
       
#       rgb = raw.postprocess()


