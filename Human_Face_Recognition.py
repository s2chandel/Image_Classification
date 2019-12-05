# Face Recognition Model
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


model = Sequential()
# Convolutiion layer
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
# 2ndlayer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
# 3rd layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
# 4th layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
# 5th layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
# regularizer
model.add(Dropout(0.2))
# flattening output
model.add(Flatten())
# Full connection
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation = 'softmax')) #(dense(classes,))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


model.summary()


# layers
for layer in classifier.layers:
    print(layer.output_shape)


# Part 2 - Fitting the CNN to the images

# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

# spliting images folder into train_test_validation 
import split_folders

split_folders.ratio('facial_data/training_set', output="test", seed=1337, ratio=(.8, .2)) # default values


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# training_data
# method used to identify labels if images are kept in their respective folders
!ls
training_set = train_datagen.flow_from_directory('train_set',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
# validation_data
test_set = test_datagen.flow_from_directory('test_set',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# test_arr = np.array(test_set)

history = model.fit_generator(training_set,
                         samples_per_epoch = 500,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 100)


loss, acc = model.evaluate(test_set, verbose = 0)
print(acc * 100)

# model evaluation
loss, accuracy = model.evaluate(training_set, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(test_set, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))



# model history plot(train_val_acc and train_val_loss)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.legend()

plot_history(history)



# Model Inferrence

from keras.preprocessing import image
image = image.load_img('test_images/person4.4.jpg',target_size=(500,500))
test_image =  img_to_array(image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
classes = training_set.class_indices


# chk = np.array([[0,0,0,1]])
def prediction(result):
    label = np.argmax(result, axis=-1)
    if label == 0:
        prediction = 'Person1'
    if label == 1:
        prediction = 'Person2'
    if label == 2:
        prediction = 'Person3'
    else:
        prediction = 'Person4'  
    print('====Image Recognised as====')

    return prediction


prediction(result)





