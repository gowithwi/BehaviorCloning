# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

It also include extra files, with the neural network trained on my private training set.
* model_my_noGen.h5
* model_my_noGen.py
* video_my_noGen.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains 3 epoch in order to reduce overfitting. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The training data was downloaded from Udacity website.

First, I normalized and centralized the image (-0.5 ~ 0.5).

Second, I used a combination of center lane driving, recovering from the left and right sides of the road.

Third, I flipped the images to double the dataset.

Forth, I cropped the images, using only the road part data from the camera to train the model.

Fifth, (which I think is most important here), I screened out data with steering less than 1degree.

###  5. Model Architecture and Training Strategy

The overall strategy for deriving a model architecture--NVIDIA Architecture, was as below:

```sh

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
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

* example of (1)left/right camera, where I set steering at steering_original+0.2, and (2) flipped image, where I set steering at steering_original*(-1)
```sh
https://user-images.githubusercontent.com/29034510/33233671-725ab224-d1ce-11e7-9f8e-6a12b929a013.png
```
