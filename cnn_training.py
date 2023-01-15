import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import time

NAME = 'image_cor_cnn_64x64_{}'.format(int(time.time()))

# Load training images from Dataset/Train 

print("\n\nLoading Images ...")

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

x_train = np.asarray(train_images[:,2])
x_train = np.asarray([np.squeeze(i) for i in x_train])
x_train = x_train/255
y_train = np.asanyarray(train_images[:,0:2], dtype='int64')

X_train, X_test,y_train, y_test = train_test_split(x_train,y_train ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)

print("\nNo. of Training Images:", X_train.shape[0])
print("No. of Validation Images:", X_test.shape[0])
print("")

def get_test_sq(img, sq_size, sq_loc, plot=False):
#     img = cv2.copyMakeBorder(img, int(sq_size[1]/2), int(sq_size[1]/2), int(sq_size[0]/2), int(sq_size[0]/2), cv2.BORDER_CONSTANT, None)

    (x,y) = (sq_loc[0]-int(sq_size[0]/2),sq_loc[1]-int(sq_size[1]/2))
    crop = img[y:y+sq_size[0], x:x+sq_size[1]]
    if plot:
        sq = img.copy()
        sq = cv2.rectangle(sq, (x,y), (x+sq_size[0], y+sq_size[1]), (255,0,0), 2)
        fig, ax = plt.subplots(1,2, figsize=(10,10), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(sq)
        ax[1].imshow(crop)
    return crop

X_train = X_train.reshape((-1,64,64,1))
X_test = X_test.reshape((-1,64,64,1))

data_aug = keras.Sequential([layers.experimental.preprocessing.RandomRotation(factor=(-0.25, 0.25), input_shape=(64,64,1)), 
                             layers.experimental.preprocessing.RandomFlip()])

# CNN Architecture

cnn_model = keras.Sequential([
    
    data_aug,
    
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPool2D((2,2)),
    
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(2, activation='relu')
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME), histogram_freq=1)

cnn_model.compile(optimizer='adam', loss='mean_absolute_error')
print(cnn_model.summary())

# Start Training 

print("")
epochs = int(input("Enter no. of epochs: "))
print("")
print("Starting Training ...")
cnn_model.fit(X_train, y_train, epochs=epochs, callbacks=[tensorboard_callback])

print("Done Training.\n")

print("Starting Validation ...")
cnn_model.evaluate(X_test,y_test)
print("Done Validation.\n")