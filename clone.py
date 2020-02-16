# clone_v2.4: Improving generator function
import csv
import cv2
import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from extra import generator

# Define this file version
VERSION = 2.4

""" Load training data and split it into training and validation set """
def load_samples(data_path):
    samples = []
    # Read csv file
    with open(os.path.join(data_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    # Remove column names
    samples.pop(0)
    # Split samples to get the train set
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples,validation_samples

# Define Nvidia neural network
def Nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66,200,3)))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model
    
# Train model with calling the generator function
def train_model(model, train_samples, validation_samples ,data_folder):
    batch_size = 40
    learning_rate = 0.0001
    validation_steps = np.ceil(len(validation_samples)/batch_size)
    steps_per_epoch = 400
    nb_epoch = 10
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    train_generator = generator(data_folder, train_samples, batch_size, True)
    validation_generator = generator(data_folder, validation_samples, batch_size, False)
    model.fit_generator(train_generator, 
                steps_per_epoch=steps_per_epoch, 
                validation_data=validation_generator, 
                validation_steps=validation_steps, 
                epochs=nb_epoch, verbose=1)
    # Save model
    model.save('model_v'+str(VERSION)+'_'+ data_folder +'.h5')
if __name__ == '__main__':
    # Define data to use
    data_folder = 'data2'
    # Load data sets
    train_samples,validation_samples = load_samples(data_folder)
    # Define neural network model
    model = Nvidia_model()
    # Train model and save it
    train_model(model, train_samples, validation_samples, data_folder)