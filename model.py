import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                
                name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name_left)
                
                name_right = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name_right)

                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3])+0.2
                right_angle = float(batch_sample[3])-0.2                
                
                if center_angle >0.1 or center_angle < -0.1 :
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(cv2.flip(center_image,1))                
                    angles.append(center_angle*-1.0)
                    
                if left_angle >0.1 or left_angle < -0.1 :              
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(cv2.flip(left_image,1))                
                    angles.append(left_angle*-1.0)
                    
                if right_angle >0.1 or right_angle < -0.1 :              
                    images.append(right_image)                
                    angles.append(right_angle)
                    images.append(cv2.flip(right_image,1))                
                    angles.append(right_angle*-1.0)
              

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=4)
validation_generator = generator(validation_samples, batch_size=4)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model_gen_udata.h5')









