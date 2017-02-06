# Udacity Self Driving Car Nanodegree - Project 3



### Project Objective
Build and train a neural network to learn to steer a car on the Udacity Simulator.


### Summary
The results of this project are that the Udacity Simulator car can successfully drive itself around track 1. The limitations discovered in using the Keras ImageDataGenerator have shed invaluable insight into using Convolutional Neural Networks to solve regression problems.


### Files Submitted & Code Quality

My project includes the following files:

    model.py - contains the script to create and train the model
    drive.py - for driving the car in autonomous mode
    model.json - contains a trained convolution neural network
    READMER.md - summarizes the results

Using the Udacity provided simulator and drive.py, the car can be driven autonomously around the track by executing

*python drive.py model.json*

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.


### Data Processing
The Udacity dataset was used for training. It was recorded on track 1 of the Udacity Simulator. Additional data was not generated because steering the car with the keyboard produces discontinuous turning angle data. The data consists pre-dominantely of zero turn data for the center camera. Images for the left and right cameras are provided, but without a turning angle.

In order to ensure that the dataset is properly organized, the correct image and it's 'center camera' turning angle are matched together. The turning angle is then increased or decreased by 0.25 and matched to the left or right camera images respectively. This results in a data distribution seen in the *Before Augmentation* figure below. The number of -0.25, 0 and 0.25 turning angle images are the result of shifting the 0 turn angle by +- 0.25.

In order to remove the unnecessary sky and car bonnet from each image, the dataset images are cropped from 160x320 to 80x320 pixels. The images are then resized to 64x64 pixels. 

![Pre Augmentation](https://cloud.githubusercontent.com/assets/22233694/22617144/ffc5f0d8-eac5-11e6-9b6a-35898e4cb486.png "Pre Augmentation")


### Data Augmentation
The performance of the car is enhanced by providing more turning data during training. In order to do this, all images with a turning angle greater than 0.05 or less than -0.05 are flipped along the vertical axis. The corresponding turning angle is multiplied by -1 and this new data added to the dataset. The number of data points increased from 24k to 43k (78% increase).

![Post Augmentation](https://cloud.githubusercontent.com/assets/22233694/22617143/fdcb6ace-eac5-11e6-8795-4996c8b6978c.png "Post Augmentation")


### Neural Network Design
The Neural Network Design is based on the [NVIDIA network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) used in their DAVE-2 System. It has 5 Convolutional layers and 3 Fully Connected layers. The 1st, 3rd and 5th Convolutional layers are followed by Max-Pooling layers and each Fully Connected layer followed by a Dropout layer (to reduce overfitting). The network ends with a Linear Activation layer, as is normal for regression learning.

The Batch Normalization layer centers the data to zero-mean and unit variance. In training deep networks, if only the input is normalized, hidden layers gradually deviate from zero mean and unit covariance. This is know as [internal co-varaiate shift](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf). Batch Normalization is used to counter this and is commonly used either before or after the Activation layer. In this network it is used before the Activation layer.

Leaky Relu Activation layers are used to help reduce the number of ['dead' relus](http://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-2.html). This can occur if the learning rate is too high. With the Adam Optimizer, the learning rate is set at 0.001 by default and slowly decreases over time. It is therefore just a precaution to use Leaky Relus.

The network has a total of 187k parameters. This is less than the NVIDIA model which has 250k. Interestingly, it seems that increasing the depth of the network worsens performance.


| Layer | Output Shape | Parameters |
| ----- | ------------ | ---------- |
| BatchNormalization | (None, 64, 64, 3) | 6 |
| --- | --- | --- |
| Convolution2D | (None, 60, 60, 24) | 1824 |
| BatchNormalization | | 48 |
| LeakyReLU | | |
| MaxPooling2D | (None, 30, 30, 24) | |
| --- | --- | --- |
| Convolution2D | (None, 26, 26, 36) | 21636 |
| BatchNormalization | | 72 |
| LeakyReLU | | |
| Convolution2D | (None, 22, 22, 48) | 43248|
| BatchNormalization | | 96 |
| LeakyReLU | | |
| MaxPooling2D | (None, 11, 11, 48) | |
| --- | --- | --- |
| Convolution2D | (None, 9, 9, 56) | 24248 |
| BatchNormalization | | 112 |
| LeakyReLU | | |
| Convolution2D | (None, 7, 7, 64) | 32320 |
| BatchNormalization | | 128 |
| LeakyReLU | | |
| MaxPooling2D | (None, 3, 3, 64) | |
| --- | --- | --- |
| Flatten | (576) | |
| Dense | (100) | 57700 |
| BatchNormalization | | 200 |
| LeakyReLU | | |
| Dropout | 50% | |
| --- | --- | --- |
| Dense | (50) | 5050 |
| BatchNormalization | | 100 |
| LeakyReLU | | |
| Dropout | 50% | |
| --- | --- | --- |
| Dense | (10) | 510 |
| BatchNormalization | | 20 |
| LeakyReLU | | |
| Dropout | 50% | |
| --- | --- | --- |
| Dense | (1) | 11 |
| BatchNormalization | | 2 |
| Activation | 1 |  |
____________________________________________________________________________________________________
Total parameters: 187331


### Training
The model is compiled using the Adam optimizer as it has a built-in decaying learning rate. The Mean Squared Error loss function is utilized to calculate the error between the actual turning angle and the network predicted steering angle.

A Data Generator is used to prevent saturating memory with large datasets. To improve efficiency, the generator is run in parallel to the model. This allows real-time data augmentation on images on CPU in parallel to training the model on GPU. [See Keras documentation](https://keras.io/models/sequential/). 

Due to the simplicity of the Keras generator, the Keras ImageDataGenerator is used to produce the training and validation batches for this network. No augmentation is done in the ImageDataGenerator as the ealier data augmentation is sufficient. There are two very important negative consequences to choosing the Keras ImageDataGenerator over a custom generator. They are:

- Certain image augmentation functions cannot be done with the Keras ImageDataGenerator. Amongst them are random image brightening, horizontal shift and horizontal flip. This is because the turning angle does not change when an image is altered, resulting in an INCORRECT turning angle associated with the augmented image. This will negatively train the model.
- The input and validation datasets are split (ratio 85:15) outside the generator. The training and validation data are now fixed and will not be shuffled together inbetween epochs. This means that if during each epoch the network is trained on the entire input dataset and is validated on the entire validation dataset, it will see exactly the same data every epoch. This is terrible for generalizing a network. To overcome this, each epoch ends after being trained and validated on 50% of the images in the training and validation datasets respectively. The datasets are then shuffled and 'new' data is trained and validated on.

The model was not updated to use a custom generator as it is possible to achieve the goal of this project with the Keras generator. The lessons learnt above are also invaluable and using a custom generator would not have brought them to light. The model also allows the car to complete most of the 2nd track on the 'Fastest' quality setting. It however cannot overcome the shadows in higher quality runs. 


### Testing
The model is tested on the Udacity Simulator, tracks 1 and 2. The SCD Nanodegree only evaluates the performance of the car on track 1. 

The model successfully steers the car around track 1 on all screen resolutions and all graphics quality settings. It is expected that the car attempts to avoid the shadows on the road as no 'shadow image augmentation' was done.

Track 2 was completely unseen during training and provides for an interesting test of how generalized the model is. On the lowest quality setting (no shadows) the car almost makes it almost to the end of the track, getting stuck on a tight right hand corner before the steep hill. It steers succesfully through all of the previous corners.


### Future Improvements
Due to the lessons learnt and short time available to complete this project (approx 10 days), there are a few simple improvements that can be done to improve the generalization of the model. They are:
- Balance the dataset so that there is no spike in 0, =0.25 and -0.25 data.
- Use a custom generator
- Use data augmentation in the custom generator, eg shadows, brightness, image shift and horizontal flip
