# CarND_P3-Behavioral-Cloning
Udacity Self Driving Car Nanodegree - Project 3


### Project Objective
Build and train a neural network to learn to steer a car on the Udacity Simulator.


### Summary
The results of this project are that the Udacity Simulator car can successfully drive itself around track 1. The limitations discovered in using the Keras ImageDataGenerator have shed invaluable insight into using Convolutional Neural Networks to solve regression problems.


### Data Organization
The Udacity dataset was used for training. Additional data was not generated because steering the car with the keyboard produces discontinuous turning angle data. 

In order to ensure that the dataset is properly organized, the correct image and it's 'center camera' turning angle are matched together. The turning angle is increased or decreased by 0.25 depending on whether the image is from the left or right camera respectively. In order to remove the unnecessary sky and car bonnet, the dataset images were cropped from 160x320 to 80x320 pixels. The images were then resized to 64x64 pixels and saved in a pickled file for ease of loading.


### Data Augmentation
The Udacity dataset was recorded on track 1 of the Udacity Simualtor in an anti-clockwise direction only. This results in a majority of straight and left turn data. In order to balance the dataset, all images with a turning angle greater than 0.07 or less than -0.07 are flipped along the vertical axis. The corresponding turning angle is multiplied by -1 and this new data added to the dataset. The number of data points increased from 24k to 42k (75% increase). This also assured a balanced learning between left and right turns.


### Neural Network Design
The Neural Network Design is based on the [NVIDIA network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) used in their DAVE-2 System. It has 5 Convolutional layers and 3 Fully Connected layers. The 1st, 3rd and 5th Convolutional layers are followed by Max-Pooling layers and each Fully Connected layer followed by a Dropout layer. The network ends with a Linear Activation layer, as is normal for regression learning.

The BatchNormalization layer centers the data to zero-mean and unit variance. In training deep networks, if only the input is normalized, hidden layers gradually deviate from zero mean and unit covariance. This is know as [internal co-varaiate shift](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf). Batch Normalization is used to counter this and is commonly used either before or after the Activation layer. In this network it is used before the Activation layer.

Leaky Relu Activation layers are used to help reduce the number of ['dead'](http://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-2.html) relus. This can occur if the learning rate is too high. With the Adam Optimizer, the learning rate is set at 0.001 by default and slowly decreases over time. It is therefore just a precaution to use Leaky Relus.

The network has a total of 187k parameters. This is less than the NVIDIA model which has 250k. Interestingly, it seems that increasing the depth of the network worsens performance.



BatchNormalization - (None, 64, 64, 3)
____________________________________________________________________________________________________
Convolution2D - (None, 60, 60, 24)

BatchNormalization

LeakyReLU

MaxPooling2D - (None, 30, 30, 24)
____________________________________________________________________________________________________
Convolution2D - (None, 26, 26, 36)

BatchNormalization

LeakyReLU
____________________________________________________________________________________________________
Convolution2D - (None, 22, 22, 48)

BatchNormalization

LeakyReLU

MaxPooling2D - (None, 11, 11, 48)
____________________________________________________________________________________________________
Convolution2D - (None, 9, 9, 56)

BatchNormalization

LeakyReLU
____________________________________________________________________________________________________
Convolution2D - (None, 7, 7, 64)

BatchNormalization

LeakyReLU

MaxPooling2D - (None, 3, 3, 64)
____________________________________________________________________________________________________
Flatten - (576)

Dense - (100)

BatchNormalization

LeakyReLU

Dropout - (50%)
____________________________________________________________________________________________________
Dense - (50)

BatchNormalization

LeakyReLU

Dropout - (50%)
____________________________________________________________________________________________________
Dense - (10)

BatchNormalization

LeakyReLU

Dropout - (50%)
____________________________________________________________________________________________________
Dense - (1)

BatchNormalization

Activation - (1)
____________________________________________________________________________________________________
Total parameters: 187331


### Training
The model is compiled using the Adam optimizer as it has a built-in decaying learning rate. The Mean Squared Error loss function is utilized to calculate the error between the actual turning angle and the network predicted steering angle.

A Data Generator is used to prevent saturating memory with large datasets. To improve efficiency, the generator is run in parallel to the model. This allows real-time data augmentation on images on CPU in parallel to training the model on GPU [See Keras documentation](https://keras.io/models/sequential/). 

Due to the simplicity of the Keras generator, the Keras ImageDataGenerator is used to produce the training and validation batches for this network. No augmentation is done in the ImageDataGenerator as the data balancing is sufficient. There are, however, two very important negative consequences to choosing the Keras ImageDataGenerator over a custom generator. They are:

- Certain image augmentation functions cannot be done with the Keras ImageDataGenerator. Amongst them are random image rotation, horizontal shift and horizontal flip. This is because the turning angle does not change when an image is altered, resulting in an INCORRECT turning angle associated with the augmented image. This will negatively train the model.
- The input and validation datasets are split (ratio 85:15) outside the generator. The training and validation data are now fixed and will not be shuffled together inbetween epochs. This means that if during each epoch the network is trained on the entire input dataset and is validated on the entire validation dataset, it will see exactly the same data every epoch. This is terrible for generalizing a network. To overcome this, each epoch ends after being trained on 50% of the images and validated on 33% of the validation set. The datasets are then shuffled and 'new' data is trained and validated on. This can be very limiting as certain beneficial data augmentation methods cannot be used (eg random image brightening, horizontal flips, horizontal shifts, shadows).

The model was not updated to use a custom generator as it is possible to achieve the goal of this project with the Keras generator. The lessons learnt above are also invaluable and using a custom generator would not have brought them to light. The model also allows the car to complete most of the 2nd track on the 'Fastest' quality setting. It however cannot overcome the shadows in higher quality runs. 


### Testing
The model is tested on the Udacity Simulator, tracks 1 and 2. The SCD Nanodegree only evaluates the performance of the car on track 1. 

The model successfully steers the car around track 1 on all screen resolutions and all graphics quality settings. It is expected that the car attempts to avoid the shadows on the road as no 'shadow training' was done.

Track 2 was completely unseen during training and provides for an interesting test of how generalized the model is. On the lowest quality setting (no shadows) the car makes it almost to the end of the track, **************succesfully************ steering through most of the tight corners. It is unable to get up the last hill.


### Future Improvements
Due to the lessons learnt and short time available to complete this project (1 week), there are a few simple improvements that can be done to improve the generalization of the model. They are:
- Use a custom generator
- Use data augmentation in the custom generator, eg shadows, brightness, image shift and horizontal flip
