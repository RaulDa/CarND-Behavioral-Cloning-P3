import csv
import cv2
import numpy as np
import sklearn
from PIL import Image

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

# Import image paths and steering angle from driving_log
lines = []
#with open('C:/Users/Raul/Documents/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split data for training and validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generator to train the model splitting the data into batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            # For each batch, read all image paths and steering angles, process pictures and store them in Numpy array
            for batch_sample in batch_samples:
                for i in range(3):
                    # Read image path. "i" will give the corresponding camera (0 -> center, 1 -> left, 2 -> right)
                    #name = 'C:/Users/Raul/Documents/CarND-Behavioral-Cloning-P3/data/IMG/'+batch_sample[i].split('/')[-1]
                    name = '/home/carnd/data/IMG/'+batch_sample[i].split('\\')[-1]

                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to get a better contrast distribution
                    img = Image.open(name)
                    image = np.array(img)
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    lab_planes = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0)
                    lab_planes[0] = clahe.apply(lab_planes[0])
                    lab = cv2.merge(lab_planes)
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                    # Apply correction to angles of left and right cameras
                    if i == 0:
                        angle = float(batch_sample[3])
                    elif i == 1:
                        angle = float(batch_sample[3]) + correction 
                    else:
                        angle = float(batch_sample[3]) - correction
						
                    # Store both image and angle
                    images.append(image)
                    angles.append(angle)

            # Data augmentation. Each stored image is flipped to avoid turn left bias.
            # Opposite sign for steering angle of flipped image is taken
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
	
            # Return numpy array of images and measurements
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Definition of correction factor for images of left and right cameras
correction = 0.2

# Apply generator for both training and validation data
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# CNN for training. Data is previously normalized and cropped (irrelevant regions filtered)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use of Mean Squared Error and Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Training and validation (20% dataset for validating). Shuffle activated
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6, validation_data = validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=8)

model.save('model.h5')
