
# coding: utf-8

# In[25]:

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas

import pickle
import csv
import matplotlib.pyplot as plt
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
import cv2
import numpy as np
import pandas as pd
import h5py
import sys

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

## based on vivek's approach for artificially generating the situations
def trans_image(image,steer,trans_range = 50):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(128,128))
    
    return image_tr,steer_ang

def augment_brightness(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# In[26]:

##Read in the csv containing immage paths and associated steering angles 
lines = []

with open('data/driving_log.csv') as f:
    lines = pandas.read_csv(f, header=0, skipinitialspace=True).values

images = []
angles = []

outputsize = (128,128)
i =0
for line in lines:

    #read and augment center camera images
    name = 'data/IMG/'+line[0].split('/')[-1]
    center_image = cv2.resize(cv2.imread(name)[55:135, :, :],outputsize,interpolation = cv2.INTER_AREA)
    center_image = augment_brightness(cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB))
    center_angle = float(line[3])
    center_image,center_angle = trans_image(center_image,center_angle)
    images.append(center_image)
    angles.append(center_angle)

    center_image = cv2.flip(center_image,1)
    center_angle = -center_angle
    center_image,center_angle = trans_image(center_image,center_angle)
    images.append(center_image)
    angles.append(center_angle)

    #read and augment left camera images
    lname = 'data/IMG/'+line[1].split('/')[-1]
    left_image =cv2.resize(cv2.imread(lname)[55:135, :, :],outputsize,interpolation = cv2.INTER_AREA)
    left_image = augment_brightness(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    left_angle = float(line[3]) + 0.2
    left_image,left_angle = trans_image(left_image,left_angle)
    images.append(left_image)
    angles.append(left_angle)


    left_image = cv2.flip(left_image,1)
    left_angle = -left_angle
    left_image,left_angle = trans_image(left_image,left_angle)                
    images.append(left_image)
    angles.append(left_angle)

    #read and augment right camera images
    rname = 'data/IMG/'+line[2].split('/')[-1]
    right_image = cv2.resize(cv2.imread(rname)[55:135, :, :],outputsize,interpolation = cv2.INTER_AREA)
    right_image = augment_brightness(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    right_angle = float(line[3]) - 0.2
    right_image,right_angle = trans_image(right_image,right_angle)               
    images.append(right_image)
    angles.append(right_angle)


    right_image = cv2.flip(right_image,1)
    right_angle = -right_angle
    right_image,right_angle = trans_image(right_image,right_angle)               
    images.append(right_image)
    angles.append(right_angle)

    i = i + 1
    
    if i % 1000 == 0:
        print(i/1000)


# In[27]:

#visualize images
plt.imshow(images[20])
plt.show()

print(images[0].shape)


# In[28]:


# check the distribution of images
plt.hist(angles,bins=100)
plt.show()


#Convert the image data to numpy array
def as_array(array):
    return (np.array(array))

x_train = np.array(images)
y_train = np.array(angles)


# In[29]:



print("Training!")
model=Sequential()
model.add(Lambda(lambda x: x/255 -0.5,input_shape=(128,128,3)))
model.add(Convolution2D(3, 3,3))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(5, 3,3))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse' , optimizer = 'adam')

model.fit(x_train, y_train, validation_split = 0.25, shuffle = True, epochs = 4)

model.save("model.h5") #change model name according to the data combinations
print("Model saved")

