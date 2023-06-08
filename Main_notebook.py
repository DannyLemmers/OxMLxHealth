#!/usr/bin/env python
# coding: utf-8
Lets start with importing some stuff. We will use imageio to read the images.
# In[1]:


import imageio as iio
import re
import os
from os import listdir
from os.path import isfile, join



# In[3]:


def getImageID(imageList):
    ID = []
    for i in imageList:
        ID.append()



localPath = os.getcwd()
dataPath = localPath +"\data"

images = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
print(images[0])

IDs = getImageID(images)
#img = iio.imread()


# In[ ]:




