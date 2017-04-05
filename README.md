# **Behavioral Cloning** 
### Self-Driving Car Engineer Nanodegree - _Project 3_
### By: **Soroush Arghavan**
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./0.jpg "Hard Swerve Enter"
[image2]: ./1.jpg "Hard Swerve"
[image3]: ./2.jpg "Hard Swerve Recovery"
[image4]: ./3.jpg "Soft Swerve Enter"
[image5]: ./4.jpg "Soft Swerve"
[image6]: ./5.jpg "Soft Swerve Recovery"
[image7]: ./7.jpg "Cropped Image"
[image8]: ./6.png "Losses"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous model
* model.h5 containing a trained convolution neural network 
* video.mp4 demonstrating the simulation finishing a successful lap autonomously
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture used in this project is based on the model presented by [Bojarski et al.](https://arxiv.org/pdf/1604.07316v1.pdf). Slight adjustments have been made in order to optimize the model for this project.
Firstly, the input of the model is normalized in line 64. Furthermore, the top 70 and the bottom 25 pixels of the input are cropped in line 66, in order for the model to be trained only on the area of interest.
Also, in line 75, a dropout layer with a probability of 50% has been added to prevent overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 75). Furthermore, RELU activation layers were added to the CNN in order to add nonlinearity to the model and further reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The following strategies were used in data collection:
* At least three laps of data were recorded to ensure ample training and validation data sizes
* One of the three laps was done clockwise to combat the bias and oversteering to the left.
* Some segments were completed on straight lines with minimal steering input, focusing on staying at the center of the road
* Some segments were completed while gently swerving towards the edges of the road and then recovering and moving towards the other edge. By swerving, the model could learn from the training data how to recover from the edges in high radius turns.
* Finally, some segments were completed while swerving aggressively towards the edges of the road and then recovering and moving towards the other edge. The hypothesis was that the model can learn from the aggressive turning how to handle low radius turns.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to optimize the NVidia CNN for the current project.

My first step was to use the model without additional modification, with cropped inputs. This model, trained with the images from all three cameras and with 2 epochs only, proved to be able to complete a lap without running out of the track when trained using the sample data provided by Udacity.

Next, I collected my own training data using the strategies mentioned in Section 4. The model was not able to complete the first turn using my training data.

Looking at the number of epochs and the trend in training and validation losses, it was clear that the model could be further trained for a number of epochs. Although the validation loss seems to be slightly unstable, a decreasing trend is visible and 10 epochs seems to be enough for convergence.

The model behavior was improved. However, the car would still go off-track when the edges of the road were unclear. This can be a result of overfitting. To combat this, a dropout layer with a probability of 0.5 was added to the model.

The new model was able to successfully complete a lap around track 1.

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-79) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 3-channel image						| 
| Normalizing        	| 												|
| Cropping				| Top 70 and bottom 25 rows						|
| Convolution 5x5     	| 2x2 stride, depth 24						 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, depth 36						 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, depth 48						 	|
| RELU					|												|
| Convolution 3x3     	| depth 64						 				|
| RELU					|												|
| Convolution 3x3     	| depth 64						 				|
| RELU					|												|
| Flattern   			|  	        									|
| Fully connected		| outputs 100       							|
| Fully connected		| outputs 50        							|
| Fully connected		| outputs 10        							|
| Fully connected		| outputs 1			 							|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded at least three laps of data. Two laps were recorded while driving counter clockwise and one was recorded while driving clockwise to prevent bias.

During the three laps two tactics were used to collect strong enough data for the model to train on. By swerving sharply and gently towards the edges of the road and then recovering from the swerve, the goal was to collect steering data for situations when the car is close to the lane so that the situation can be later associated with a counter-acting steering value in the weights of the model.

A hard swerve motion which aims to be used to train the car for cornering is displayed below. 
![alt text][image1]
![alt text][image2]
![alt text][image3]

A soft swerve which can be used for straight roads where the car needs to maintain a center position can be seen below.

![alt text][image4]
![alt text][image5]
![alt text][image6]

The data from the three laps and the three cameras were used for model training. The images were not preprocessed. The data was shuffled and split into training and validation sets with a 4:1 ratio.

One of the Keras layers added was a Cropping layer which crops the top 70 and the bottom 25 pixels of the images in order to focus on the road section only. A sample of the region of interest can be seen below.

![alt text][image7]

The training and validation losses both decreased with 10 epochs which did not raise any concerns. The losses can be seen below.

![alt text][image8]

