# v5
import csv
import cv2
import numpy as np

lines = []
with open('./TestData7/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for  line in reader:
                lines.append(line)
images = []
measurements = []
m = -2.0

for line in lines:
    if m>0:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = './TestData7/IMG/' + filename
            image = cv2.imread(current_path)
#            images.append(image)
            m1 = float(i)
            m2 = m1*(-0.3*m1 + 0.5)
            measurement = float( float(line[3]) + m2)
            if measurement > 0.1:
                images.append(image)
                measurements.append(measurement)
            if measurement<-0.1:
                images.append(image)
                measurements.append(measurement)

    m = 2.0

augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
        
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#X_train = np.array(images)
#y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build the Final Test Neural Network in Keras Here
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

model.compile(loss='mse',optimizer='adam')
#model.fit(X_train,y_train, validation_split =0.2,shuffle=True,nb_epoch=7)
model.fit(X_train,y_train, validation_split =0.1,shuffle=True,nb_epoch=5)

model.save('model_test_6.h5')
#exit()