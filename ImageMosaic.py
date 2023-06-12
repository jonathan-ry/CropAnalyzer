'''
Driver script. Execute this to perform the mosaic procedure.
'''
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout, QLineEdit)
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QDir
from PyQt5.QtCore import QProcess
# import numpy as np  
from subprocess import call
import time
import utilities as util
import Combiner
import cv2
import Dataset
import os
import datetime
import Perspective
import shutil
start_time = time.time()
from msg_box import Ui_MainWindow

now = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')

Dataset.write()

if os.path.isdir('results') == True:
    os.rename('results', 'results - ' + str(now))

os.mkdir('results')

fileName = "datasets/imageData.txt"
with open('imstitchpath.txt','r') as f:
    imageDirectory= f.read()
# imageDirectory = "datasets/images/"
# print(imageDirectory)
imageDirectory = imageDirectory + "/"

print ("Creating Temp Directory")

if os.path.isdir('temp') == True:
    shutil.rmtree('temp', ignore_errors=False, onerror=None)

os.mkdir('temp')

print ("Copying Images to Temp Directory")

allImages, dataMatrix = util.importData(fileName, imageDirectory)
Perspective.changePerspective(allImages, dataMatrix)

print ("Stitching Images")

result = Combiner.combine()

util.display("RESULT", result, 4000)
cv2.imwrite("results/finalResult.jpg", result)

print ("Done. Find your final image in results folder as finalResult.png")
print("Process finished --- %s seconds ---" % (time.time() - start_time))
# window = QtWidgets.QMainWindow()
# msg_box = Ui_MainWindow()
# msg_box.setupUi(self.window)
# window.show()
# def msg(self):
#     self.window = QtWidgets.QMainWindow()
#     self.msg_box = Ui_MainWindow()
#     self.mesg_box.setupUi(self.window)
#     self.window.show()