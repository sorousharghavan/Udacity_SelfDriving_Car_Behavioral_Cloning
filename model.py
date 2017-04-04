import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

#Generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #Training teering input
                steering_center = float(batch_sample[3])
                
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                #Add all three images to the list
                images.append(cv2.imread(batch_sample[0].strip()))
                images.append(cv2.imread(batch_sample[1].strip()))
                images.append(cv2.imread(batch_sample[2].strip()))
                #Add corresponding steering angles to the list
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#Read the driving log
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Split the data into a training set and a validation set
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # image format

#Model architecture

model = Sequential()
#Normalizing
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, 3)))
#Crop the top and bottom of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))
#NVidia CNN (End to End Learning for Self-Driving Cars, Bojorski et al., https://arxiv.org/pdf/1604.07316v1.pdf)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
#Added dropout layer to reduce the chance of overfitting. The rest of the CNN is the same as the NVidia CNN
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#Using Adam Optimizer
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10, verbose=1)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

exit()
