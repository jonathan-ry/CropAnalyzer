# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import PyQt5
from PyQt5 import (QtCore, QtGui, QtWidgets)
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 

from ndvi import Ui_improcessWindow
from image_stitching import Ui_stitchWindow
from machine_learning import Ui_machineWindow


class Ui_MainWindow(object):

    def imagestitching(self):
        self.window = QtWidgets.QMainWindow()
        self.machine_learning = Ui_stitchWindow()
        self.machine_learning.setupUi(self.window)
        self.window.show()

    def machinelearning(self):
        self.window = QtWidgets.QMainWindow()
        self.machine_learning = Ui_machineWindow()
        self.machine_learning.setupUi(self.window)
        self.window.show()

    def imageprocessing(self):
        self.window = QtWidgets.QMainWindow()
        self.image_processing = Ui_improcessWindow()
        self.image_processing.setupUi(self.window)
        self.window.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(856, 537)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1, 0))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("image: url(:/newPrefix/main_bg.png);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("image: url(:/newPrefix/Software/image_processing/process_bg.png);")
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.frame)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout.addLayout(self.verticalLayout_3, 2, 1, 1, 1)
        self.machineButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.machineButton.sizePolicy().hasHeightForWidth())
        self.machineButton.setSizePolicy(sizePolicy)
        self.machineButton.setMinimumSize(QtCore.QSize(0, 350))
        self.machineButton.clicked.connect(self.machinelearning)
        self.machineButton.setStyleSheet("QPushButton {\n"
"    image: url(:/newPrefix/machine_button.png);\n"
"    background-color: rgb(166, 197, 103);\n"
"    border-width: 2px;\n"
"    border-style: outset;\n"
"    border-radius: 10px;\n"
"    border-color: black;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:hover {\n"
"    image: url(:/newPrefix/machine_button.png);\n"
"    background-color: rgb(176, 197, 143);\n"
"}\n"
"QPushButton:pressed {\n"
"    image: url(:/newPrefix/machine_button.png);\n"
"    background-color: rgb(151, 197, 76);\n"
"}\n"
"\n"
"\n"
"")
        self.machineButton.setText("")
        self.machineButton.setObjectName("machineButton")
        self.gridLayout.addWidget(self.machineButton, 1, 3, 1, 1)
        self.stitchButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stitchButton.sizePolicy().hasHeightForWidth())
        self.stitchButton.setSizePolicy(sizePolicy)
        self.stitchButton.setMinimumSize(QtCore.QSize(0, 350))
        self.stitchButton.clicked.connect(self.imagestitching)
        self.stitchButton.setStyleSheet("QPushButton {\n"
"    image: url(:/newPrefix/image stitch button.png);\n"
"    background-color: rgb(166, 197, 103);\n"
"    border-width: 2px;\n"
"    border-style: outset;\n"
"    border-radius: 10px;\n"
"    border-color: black;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:hover {\n"
"    image: url(:/newPrefix/image stitch button.png);\n"
"    background-color: rgb(176, 197, 143);\n"
"}\n"
"QPushButton:pressed {\n"
"    image: url(:/newPrefix/image stitch button.png);\n"
"    background-color: rgb(151, 197, 76);\n"
"}\n"
"\n"
"\n"
"")
        self.stitchButton.setText("")
        self.stitchButton.setObjectName("stitchButton")
        self.gridLayout.addWidget(self.stitchButton, 1, 1, 1, 1)
        self.ndviButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ndviButton.sizePolicy().hasHeightForWidth())
        self.ndviButton.setSizePolicy(sizePolicy)
        self.ndviButton.setMinimumSize(QtCore.QSize(0, 350))
        self.ndviButton.clicked.connect(self.imageprocessing)
        self.ndviButton.setStyleSheet("QPushButton {\n"
"    image: url(:/newPrefix/ndvi_button.png);\n"
"    background-color: rgb(166, 197, 103);\n"
"    border-width: 2px;\n"
"    border-style: outset;\n"
"    border-radius: 10px;\n"
"    border-color: black;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:hover {\n"
"    image: url(:/newPrefix/ndvi_button.png);\n"
"    background-color: rgb(176, 197, 143);\n"
"}\n"
"QPushButton:pressed {\n"
"    image: url(:/newPrefix/ndvi_button.png);\n"
"    background-color: rgb(151, 197, 76);\n"
"}\n"
"\n"
"\n"
"")
        self.ndviButton.setText("")
        self.ndviButton.setObjectName("ndviButton")
        self.gridLayout.addWidget(self.ndviButton, 1, 2, 1, 1)
        self.moreButton = QtWidgets.QPushButton(self.centralwidget)
        self.moreButton.setMinimumSize(QtCore.QSize(20, 55))
        self.moreButton.setStyleSheet("QPushButton {\n"
"    image: url(:/newPrefix/more info.png);\n"
"    background-color: rgb(166, 197, 103);\n"
"    border-width: 2px;\n"
"    border-style: outset;\n"
"    border-radius: 25px;\n"
"    border-color: black;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:hover {\n"
"    image: url(:/newPrefix/more info.png);;\n"
"    background-color: rgb(176, 197, 143);\n"
"}\n"
"QPushButton:pressed {\n"
"    image: url(:/newPrefix/more info.png);\n"
"    background-color: rgb(151, 197, 76);\n"
"}\n"
"\n"
"\n"
"")
        self.moreButton.setText("")
        self.moreButton.setObjectName("moreButton")
        self.gridLayout.addWidget(self.moreButton, 2, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RICE TECH"))
from main_window import try_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()        
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showMaximized()
    sys.exit(app.exec_())
