#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout, QLineEdit)
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QDir
from PyQt5.QtCore import QProcess
# import numpy as np  
from subprocess import call
from prediction_msg_box import Ui_predictionWindow
from machine_learning import Ui_machineWindow


# In[2]:


data = pd.read_csv('dataset_v4.csv')


# In[3]:


data.head()


# In[4]:


data = data.drop(['ID'], axis = 1)


# In[5]:


data.info()


# In[6]:


#handling missing data
data = data.dropna()
data.info()


# In[7]:


data.describe()


# In[8]:


data['YEAR'].value_counts()


# In[9]:


data['SEASON'].value_counts()


# In[10]:


data['IRRIGATION_TYPE'].value_counts()


# In[11]:


data['MUNICIPALITY'].value_counts()


# In[12]:


data['PROVINCE'].value_counts()


# In[13]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

YEAR = le.fit_transform(data.YEAR)
SEASON = le.fit_transform(data.SEASON)
IRRIGATION_TYPE = le.fit_transform(data.IRRIGATION_TYPE)
MUNICIPALITY = le.fit_transform(data.MUNICIPALITY)
PROVINCE = le.fit_transform(data.PROVINCE)
data['YEAR'] = YEAR
data['SEASON'] = SEASON
data['IRRIGATION_TYPE'] = IRRIGATION_TYPE
data['MUNICIPALITY'] = MUNICIPALITY
data['PROVINCE'] = PROVINCE



# In[14]:


print(data['YEAR'].unique())
print(data['SEASON'].unique())
print(data['IRRIGATION_TYPE'].unique())
print(data['MUNICIPALITY'].unique())
print(data['PROVINCE'].unique())


# In[15]:


data.head()


# In[16]:


print(data)


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[20]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error, r2_score
forest = RandomForestRegressor(n_estimators=1000,
                              criterion='mse',
                              random_state=1,
                              n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' %(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[21]:


print(forest.score(X_test, y_test))


# In[22]:


forest.predict(X_test)


# In[23]:


X_test


# In[ ]:


# YEAR = input('Enter Year:')
# MONTH = input('Enter Month:')
# SEASON = input('Enter Season:')
# SIZE = input('Enter Size:')
# AVG_TMAX = input('Enter Temperature Max:')
# AVG_TMIN = input('Enter Temperature Min:')
# AVG_HUM = input('Enter Humidity:')
# N = input('Enter Nitrogen Composition:')
# P = input('Enter Phosporus Composition:')
# K = input('Enter Potassium Composition:')
# IRRIGATION_TYPE = input('Enter Irrigation Type:')
# MUNICIPALITY = input('Enter Municipality:')
# PROVINCE = input('Enter Province:')

with open('town.txt', 'r') as f:
    MUNICIPALITY = f.read()
os.remove('town.txt')

with open('province.txt', 'r') as f:
    PROVINCE = f.read()
os.remove('province.txt')

with open('year.txt', 'r') as f:
    YEAR = f.read()
os.remove('year.txt')

with open('season.txt', 'r') as f:
    SEASON = f.read()
os.remove('season.txt')

with open('irrigation.txt', 'r') as f:
    IRRIGATION_TYPE = f.read()
os.remove('irrigation.txt')

with open('size.txt', 'r') as f:
    SIZE = f.read()
os.remove('size.txt')

with open('tmax.txt', 'r') as f:
    AVG_TMAX = f.read()
os.remove('tmax.txt')

with open('tmin.txt', 'r') as f:
    AVG_TMIN = f.read()
os.remove('tmin.txt')

with open('thum.txt', 'r') as f:
    AVG_HUM = f.read()
os.remove('thum.txt')

with open('phos.txt', 'r') as f:
        P = f.read()

with open('nit.txt', 'r') as f:
        N = f.read()

with open('pot.txt', 'r') as f:
        K= f.read()

with open('month.txt', 'r') as f:
        MONTH = f.read()

out_1 = forest.predict([[float(YEAR),
                        float(MONTH),
                        float(SEASON),
                        float(SIZE),
                        float(AVG_TMAX),
                        float(AVG_TMIN),
                        float(AVG_HUM),
                        float(N),
                        float(P),
                        float(K),
                        float(IRRIGATION_TYPE),
                        float(MUNICIPALITY),
                        float(PROVINCE)]])
print('Crop yield Production:', out_1)
print('Production/Sack:', out_1/50, 'sack/s')

cropy = int(out_1)
cropyield = str(cropy)
with open('crop_yield.txt', 'w') as f:
    f.write(cropyield + "kilograms")

prod = int(out_1/50)
product = str(prod)
with open('production.txt', 'w') as f:
    f.write(product + "sack/s")


window = QtWidgets.QMainWindow()
msg_box = Ui_predictionWindow()
msg_box.setupUi(window)
window.show()




# In[ ]:




