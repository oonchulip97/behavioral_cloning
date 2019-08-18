import csv
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import ndimage
from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32):
  """
  Return samples in batch for training and validation.
  """
  num_samples = len(samples)
  # Loop forever so generator never terminates
  while 1:
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      end = min(offset + batch_size, num_samples)
      batch_samples = samples[offset: end]      
       
      images = []
      measurements = []
      # Retrieve center, left, right images with corresponding angles
      # for generalization of data
      for sample in batch_samples:
        steering_angle = float(sample[3])

        correction = 0.2
        steering_left = steering_angle + correction
        steering_right = steering_angle - correction

        path = './user_data/IMG/'
        image_center = ndimage.imread(path + sample[0].split('/')[-1])
        image_left = ndimage.imread(path + sample[1].split('/')[-1])
        image_right = ndimage.imread(path + sample[2].split('/')[-1])

        images.extend([image_center, image_left, image_right])
        measurements.extend([steering_angle, steering_left, steering_right])

      augmented_images = []
      augmented_measurements = []
      # Retrieve flipped images for generalization of data
      for image, measurement in zip(images, measurements):
         augmented_images.append(image)
         augmented_measurements.append(measurement)
         augmented_images.append(np.fliplr(image))
         augmented_measurements.append(-measurement)

      X_batch = np.array(augmented_images)
      y_batch = np.array(augmented_measurements)
      yield sklearn.utils.shuffle(X_batch, y_batch)

# Read the driving log
samples = []
with open('./user_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Common neural network variables
batch_size = 12
row, col, ch = 160, 320, 3 # original image format

# Retrieve the generator
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Build the neural network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch))) # (160, 320, 3)
model.add(Cropping2D(cropping=((50, 20),(0, 0)))) # (90, 320, 3)
model.add(Conv2D(filters=3, kernel_size=(5, 5), strides=(2,2), activation='relu', padding='valid')) # (43, 158, 3)
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), activation='relu', padding='valid')) # (20, 77, 24)
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), activation='relu', padding='valid')) #(8, 37, 36)
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='valid')) # (6, 35, 48)
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')) # (4, 33, 64)
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # (1, 8448)
model.add(Dense(units=1164, activation='relu')) # (1164)
#model.add(Dropout(0.5))
model.add(Dense(units=100, activation='relu')) # (100)
#model.add(Dropout(0.5))
model.add(Dense(units=50, activation='relu')) # (50)
#model.add(Dropout(0.5))
model.add(Dense(units=10, activation='relu')) # (10)
#model.add(Dropout(0.5))
model.add(Dense(units=1)) # (1)

# Train the neural network
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
                    steps_per_epoch = ceil(len(train_samples)/batch_size), \
                    validation_data = validation_generator, \
                    validation_steps = ceil(len(validation_samples)/batch_size), \
                    epochs = 5, verbose = True)

# Visualize the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc = 'upper right')
plt.show()

# Save the model
model.save('model.h5')
