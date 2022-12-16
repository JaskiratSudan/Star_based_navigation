#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd


# In[2]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(40000, ), activation='relu'),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(
   optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanAbsoluteError,
    metrics=[keras.metrics.MeanAbsoluteError()],
)


# In[3]:


train_images = []
dataset = "Dataset/Train/"
for images in os.listdir(dataset):
    coordinates = images.split("(")[1].split(")")[0].split(",")
    img = (Image.open(dataset+images))
    img_array = np.asarray(img)
    img_array = img_array[:,:,0]
    img_array = np.array([coordinates[0],coordinates[1], img_array.flatten()], dtype=object)
    train_images.append(img_array)
    img.close()

train_images = np.asarray(train_images)


# In[4]:


print(train_images[0])
# plt.imshow(train_images[0][2])


# In[5]:


x_train = np.asarray(train_images[:,2])
x_train = np.asarray([np.squeeze(i) for i in x_train])
x_train = x_train/255
y_train = np.asanyarray(train_images[:,0:2])
print(x_train.shape)
print(y_train.shape)


# In[ ]:


model.fit(x_train, y_train, epochs=5)


# In[ ]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[ ]:


y_train

