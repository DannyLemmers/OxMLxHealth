#!/usr/bin/env python
# coding: utf-8
Lets start with importing some stuff.
# In[132]:


import subprocess
import csv
import re
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from sklearn.datasets import load_files  # efficiently load files

try:
    import tensorflow as tf
except:
    print("No tensorflow installed, make sure you have it installled!")

try:
    from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout
    from keras.layers import Activation, Flatten
    from keras.models import  Sequential
    from keras import Model, optimizers
except:
    print("No keras installed, installing keras right now")
    subprocess.call(['pip', 'install', "keras"])
    from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout
    from keras.layers import Activation, Flatten
    from keras.models import  Sequential
    from keras import Model, optimizers

try:
    from tensorflow.keras.applications.resnet50 import ResNet50
except:
    print("No keras.applications installed, installing keras.applications right now")
    subprocess.call(['pip', 'install', "keras_applications"])
    from tensorflow.keras.applications.resnet50 import ResNet50

try:
    import seaborn as sns
except:
    print("No seaborn installed, installing seaborn right now")
    subprocess.call(['pip', 'install', "seaborn"])
    import seaborn as sns

from keras.preprocessing import image  # for image preprocessing
import matplotlib.pyplot as plt # for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except:
    print("No plotly installed, installing plotly right now")
    subprocess.call(['pip', 'install', "plotly"])
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

from PIL import Image
print("Everything imported OK")


# We will start with two functions: One to get the ID of each image and one to find the index of each image of the training set. In this function, we have some folder named data containing the images and the csv file. We store all the images in the variable images, the IDs are the IDs of the images (the number after img_) and labels contain the id with the respective labels. trainIndex then is the images ID that correspond to the labels.csv.

# In[143]:


def getImageID(imageList):
    ID = []
    for i in imageList:
        ID.append(re.findall("\d{1,}", i))
    return ID

def getTrainIndex(IDs,labelsId):
   index = []
   for labeled in labelsId:
       for IDindex, ID in enumerate(IDs):
           if str(labeled) == ID[0]:
               index.append(IDindex)
   return index

localPath = os.getcwd()
dataPath = localPath +"\data\\"

images = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
del images[-1]

IDs = getImageID(images)
labels = pd.read_csv(dataPath+"\labels.csv")
trainIndex = getTrainIndex(IDs,labels.id)


# Great, so we have the images. Not all images are the same size however, lets find the maximum image size so we know how much we need to pad the images.

# In[162]:


imageSize = np.zeros(2)
for image in images:
   im = Image.open(dataPath+image)
   if imageSize[0] < im.size[0]:
       imageSize[0] = im.size[0]
   if imageSize[1] < im.size[1]:
       imageSize[1] = im.size[1]
print("Maximum image size will be: "+ str(imageSize))


# Now we know what the maximum size will be, lets pad all the images to our desired dimensions

# In[168]:


def add_margin(pil_img, top, right):
    width, height = pil_img.size
    new_width = width + right
    new_height = height + top
    result = Image.new(pil_img.mode, (new_width, new_height), (0, 0, 0))
    result.paste(pil_img, (0, top))
    return result

for image in images:
    im = Image.open(dataPath+image)
    dWidth =  imageSize[0].astype(int) - im.size[0]
    dHeight = imageSize[1].astype(int) - im.size[1]
    if dWidth or dHeight:
        im = add_margin(im, dHeight,dWidth)
    im.save('data_padded/'+image)

paddedImages = [f for f in listdir("data_padded") if isfile(join("data_padded", f))]


# In[169]:


print(paddedImages)


# In[ ]:




