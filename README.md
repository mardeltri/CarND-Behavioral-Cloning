# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./images/cnn-architecture.png "Model Architecture"
[image2]: ./images/center_2020_02_02_18_23_02_630.jpg "Center image"
[image3]: ./images/augmentation.png "Augmentation"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [clone.py](./clone.py) contains the script to create and train the model.
* [extra.py](./extra.py) includes additional functions.
* [drive.py](./drive.py) for driving the car in autonomous mode.
* [model_v2.4_data2.h5](./model_v2.4_data2.h5) containing a trained convolution neural network.
* [README.md](./README.md) summarizing the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive_nvidia.py](./drive_nvidia.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive_nvidia.py model_v2.4_data2.h5
```

#### 3. Submission code is usable and readable

The [clone.py](./clone.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


The network model is based on the [Nvidia Model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), which have been used to map the raw pixels from a front-facing camera to the steering commands for a self-driving car. This is, our same problem. The original model architecture has been depicted in the next figure.
![initial architecture][image1]

However, some modifications have been carried out in order to improve the model:
* Normalize data with a Keras lambda layer ([clone.py](./clone.py), line 37).
* Include ELU layers to introduce nonlinearity ([clone.py](./clone.py), lines 38-47).
* Add a dropout layer after the convolution layers to avoid overfitting  ([clone.py](./clone.py), line 43).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layers in order to reduce overfitting ([clone.py](./clone.py) lines 43), with a keep probability of 50%.

The model was trained and validated on different data sets to ensure that the model was not overfitting (data has been split in [clone.py](./clone.py), line 31). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Later a link to the video will be provided.

#### 3. Model parameter tuning

The model used an adam optimizer, but the learning rate was specified. A good value was found to be 0.0001, since validation loss kept improving later that with the default value 0.001. In addition, the batch size and the steps per epoch were adjusted to improve the model behavior.

The final values are:
* Learning rate: 0.0001.
* Batch size: 40.
* Steps per epoch: 400.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I create my own data set by driving along the track: twice counter-clockwise and twice clockwise. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the Nvidia model previously described.  

The model used by Nvidia is appropiate here given that it is a very similar problem. Where, having an input data from the car camera, the steering angle is computed.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, the Dropout layer was implemented.

The input was normalized to avoid saturation and to make it easier for the algorithm to compute the gradient.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. First, the curve after the bridge, since it is quite pronounced. This issue was solved by augmenting the set with translated images. Later, the final curve was not followed precisely, but in this case it was solved just by increasing the number of samples per epoch.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 34-51) consisted of the following layers:
* Lambda layer. Input shape 66x200x3
* Convolution layer. Convolution: 5x5, filter: 24, strides: 2x2, activation ELU.
* Convolution layer. Convolution: 5x5, filter: 36, strides: 2x2, activation ELU.
* Convolution layer. Convolution: 5x5, filter: 48, strides: 2x2, activation ELU.
* Convolution layer. Convolution: 3x3, filter: 64, strides: 1x1, activation ELU.
* Convolution layer. Convolution: 3x3, filter: 64, strides: 1x1, activation ELU.
* Dropout layer. Keep probability 50%.
* Fully connected layer. Neurons: 100.
* Fully connected layer. Neurons: 50.
* Fully connected layer. Neurons: 10.
* Fully connected layer. Neurons: 1.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center image][image2]

It was not necessary to record the vehicle recovering from the sides of the road back to center. However, the data set was augmented by implementing these two methods:
* Flip randomly
* Translate randomly. Images were translated and the steering angle value was increased/decreased by 0.002 degrees per pixel.

![Augmentation][image3]

The number of total images is a function of the batch size, the steps per epoch and the number of epochs. So, the total number of images is, in general, batch_size*steps_per_epoch*nb_epochs. In this case, it was 
160000 images. However, the number of original images was:
* Total number of samples: 6046 (100%)
* Number of training samples: 4836 (80%)
* Number of validation samples: 1210 (20%)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the evolution of training accuracy and validation accuracy.

#### 4. Result
Here's a [link to my video result in github](./Video.mp4) or in [YouTube](https://youtu.be/6bVqELG-A0E)