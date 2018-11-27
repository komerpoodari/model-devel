import os
import glob
import csv
import random

import numpy as np
import pandas as pd
import cv2

from sklearn import model_selection

from keras import backend as K
from keras import models, optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

ROOT_PATH = './'
BATCH_SIZE = 4
EPOCHS = 15

IMG_H = 600
IMG_W = 800
IMG_CHN = 3
NUM_CLASSES = 4  # for simulator data set red = 0, green = 1, yellow = 2, not_light = 3

MODEL_FILE_NAME = './sim_model.h5'

# check for GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Handy Augment function to apply random brightness to make model robust
def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.4, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


# Handy Augment function to zoom
def zoom(image):
    zoom_pix = random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/IMAGE_HEIGHT
    image = cv2.resize(image, None, fx=zoom_factor,fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    top_crop = (image.shape[0] - IMG_H)//2
    left_crop = (image.shape[1] - IMG_W)//2
    new_img = image[top_crop: top_crop+IMG_H, left_crop: left_crop+IMG_W]
    return new_img


# fuction to read image from file
def get_image(index, data, apply_augment=False):
    # Read image and appropiately traffic light color
    image = cv2.imread(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()))
    color = data['class'].values[index]

    if apply_augment:
        apply_brightness = random.randint(0, 1)
        apply_flip = random.randint(0, 1)
        apply_zoom = random.randint(0, 1)
        
        if apply_brightness == 1:
            image = random_brightness(image)
        if apply_flip == 1:
            image = cv2.flip(image, 1)
        if apply_zoom == 1:
            image = zoom(image)

    return [image, color]


# generator function to return images batchwise
def generator(data_set, apply_augment=False):
    while True:
        # Shuffle the data set
        shuffled_indices = np.random.permutation(data_set.count()[0])
        for batch in range(0, len(shuffled_indices), BATCH_SIZE):
            # slice out the current batch according to batch-size
            current_batch = shuffled_indices[batch:(batch + BATCH_SIZE)]

            # initializing the arrays, x_train and y_train
            x_train = np.empty([0, IMG_H, IMG_W, IMG_CHN], dtype=np.float32)
            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get an image and its corresponding color for an traffic light
                [image, color] = get_image(i, data_set, apply_augment)

                # Appending them to existing batch
                x_train = np.append(x_train, [image], axis=0)
                y_train = np.append(y_train, [color])
            y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

            yield (x_train, y_train)

#Define the model
def construct_model():
    model = Sequential()
    
    #Define Input layer
    #Add bottom cropping until bottom 180 the car bonnet is visible, mostly
    model.add(Cropping2D(cropping=((0, 0), (0, 0)),
                         input_shape=(IMG_H, IMG_W, IMG_CHN)))
    #Normalize the pixel values
    model.add(Lambda(lambda x: x/127.5 - 1.))
    
    #Define First Layer
    model.add(Conv2D(32, 8, strides=(4, 4), padding="same", activation='relu'))
    #Follow with Max Pooling Layer
    model.add(MaxPooling2D(2, 2))
    
    # Define second layer & a max pooling follow-up layer
    model.add(Conv2D(64, 4, strides=(2, 2), padding="same", activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    #Time to flatten add dropout to minimize overfitting
    model.add(Flatten())
    model.add(Dropout(.4))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(NUM_CLASSES))
 
    model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
    
    # Think about learning rate later, Use Adam 
    model.compile(optimizer=Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


if __name__ == "__main__":

    data_set = pd.read_csv(os.path.join('./sim_train.csv'))

    # Split data into random training and validation sets
    x_train, x_valid = model_selection.train_test_split(data_set, test_size=.2)

    train_gen = generator(x_train, False)
    validation_gen = generator(x_valid, False)

    model = construct_model()

    # checkpoint to save best weights after each epoch based on the improvement in val_loss
    checkpoint = ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,save_best_only=True, mode='min',save_weights_only=False)
    callbacks_list = [checkpoint]

    print('Simulator Model Training started....')
    
    history = model.fit_generator(
        train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE,
        epochs=EPOCHS, validation_data=validation_gen,
        validation_steps=len(x_valid)//BATCH_SIZE,
        verbose=1, callbacks=callbacks_list
    )

    # Destroying the current TF graph to avoid clutter from old models / layers
    K.clear_session()
