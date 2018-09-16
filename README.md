# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2017_12_02_10_58_14_690.jpg "Track one center"
[image3]: ./examples/center_2017_11_29_22_39_18_316.jpg "Track two center"
[image4]: ./examples/center_2017_11_29_22_54_40_991.jpg "Opposite direction"
[image5]: ./examples/center_2017_11_29_23_10_27_157.jpg "Coming back to center 1"
[image6]: ./examples/center_2017_11_29_23_10_28_303.jpg "Coming back to center 2"
[image7]: ./examples/center_2017_11_29_23_10_36_431.jpg "Coming back to center 3"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network that implements the architecture proposed by `NVIDIA`. `5` convolutional layers are used. The first three with a kernel of `5x5` (`3`), `2x2` stride and `24`, `36` and `48` filters. The last two with a kernel of `3x3`, non-strided and `64` filters (code lines `88-92`).

The model includes RELU layers to introduce nonlinearity and three fully-connected layers.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines `21`, `81-82` and `103`). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

In this case the training and validation, as well as good training data and data preprocessing, were enough to make the network works with reduced overfitting, so no regularization (dropout) was needed.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line `100`).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving on specially pronunciated curves for both tracks. It was important to do some preprocessing of the data to correctly deal with the shadows of the track two.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the architecture proposed by `NVIDIA`. Initially, as proposed in the project explanation, I used the `LeNet` architecture, but it was not enough to make the car driving the entire track 1.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the `NVIDIA` model achieved an adequate and similar loss and validation loss, so no overfitting occurs. Taking this into account I decided to not modify the model by applying regularization (dropout).

The final step was to run the simulator to see how well the car was driving around track one. I could see that it worked well with the training data provided in the project. However, I also created my own data set to learn how the driving data needs to be taken.

At the end of the process, the vehicle is able to drive autonomously around both tracks one and two without leaving the road and with my own data.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines `85-97`) consisted of a convolution neural network with five convolutional layers and three fully-connected layers (`NVIDIA`).

The first three convolutional layers have with a kernel of `5x5` (`3`), `2x2` stride and `24`, `36` and `48` filters. The last two have a kernel of `3x3`, non-strided and `64` filters (code lines `88-92`).

The three fully-connected layers have `100`, `50` and `10` neurons, respectively.

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. Additionally, the images are cropped to filter the irrelevant parts of the image (code lines `86-92`).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap for both tracks one and two using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

![alt text][image3]

I also recorded one lap for both tracks in the opposite direction, in order to avoid turn bias. For example, for the track two:


![alt text][image4]

Finally, I also recorded additional data with the car pointing outside of the road and coming back to the lane center, as well as data for the most pronunciated curves. Here is an example:

![alt text][image5]

![alt text][image6]

![alt text][image7]

For the data collection it was particularly important for me to use the mouse instead of the keyboard. Using the keyboard it was usual to collect data in curves with zero steering angle, which make the car leaving the road in this portions of the circuit.

Apart from taking one lap in the opposite direction, I also augmented the data set by flipping images and angles. In total, the data set is contained of `7150` samples for the track `1` and `13672` for the track `2`. Each sample contains a center, left and right image and the data is augmented, so the total number of images is `124932`.

As mentioned, I also used the images of the left and right cameras to train the model. I applied a correction factor to the steering angle of `0.2` for both cameras.

Taking into account that the shadows of track two made specially difficult the correct behavior of the car, I preprocessed this data by applying `CLAHE` (Contrast Limited Adaptive Histogram Equalization) to distribute the contrast better. At this point memory issues happened while executing the model in the AWS instance, so I introduced the generator to train the model with batches (size of `32`).

I finally randomly shuffled the data set and put `20%` of the data into a validation set.

I used this training data for training the model. The validation set helped determining whether the model was over or under fitting. The ideal number of epochs was `6`, since the loss started to decrease really slowly from the epoch `7`. A final validation of `0.0158` was achieved:

Epoch 1/6 --> loss: 0.0442 - val_loss: 0.0315

Epoch 2/6 --> loss: 0.0253 - val_loss: 0.0237

Epoch 3/6 --> loss: 0.0211 - val_loss: 0.0192

Epoch 4/6 --> loss: 0.0198 - val_loss: 0.0181

Epoch 5/6 --> loss: 0.0161 - val_loss: 0.0163

Epoch 6/6 --> loss: 0.0144 - val_loss: 0.0158

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Finally, the car is able to drive on both tracks with a desirable speed. The following [video](https://www.youtube.com/watch?v=QTId7ILCb34) shows the behavior for the track `2`.

