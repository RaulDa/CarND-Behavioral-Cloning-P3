import csv
import cv2
import numpy as np

lines = []
#with open('C:/Users/Raul/Documents/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
	
images = []
measurements = []
for line in lines:
    source_path = line[0]
    #filename = source_path.split('\\')[-1]
    filename = source_path.split('/')[-1]
    #current_path = 'C:/Users/Raul/Documents/CarND-Behavioral-Cloning-P3/data/IMG/' + filename
    current_path = '/home/carnd/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Training and validation (20% dataset for validating). Shuffle activated
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')