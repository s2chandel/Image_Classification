# Face Recognition Model

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


# Convolution Neural Network

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) # batch_size=32 filter 3*3 
classifier.add(MaxPooling2D(pool_size = (2, 2))) # filter matrix shape for maxpooling 2*2

# Step 3 - Flattening
classifier.add(Flatten())# Flattening for the output of the convolutional layer

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 4, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()

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
training_set = train_datagen.flow_from_directory('facial_data/train_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,)
                                                #  class_mode = 'categorical')
# validation_data
test_set = test_datagen.flow_from_directory('facial_data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,)
                                            # class_mode = 'categorical')

history = classifier.fit_generator(training_set,
                         samples_per_epoch = 500,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 100)



# model evaluation
loss, accuracy = classifier.evaluate(train_set, test_set, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
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




from keras.preprocessing import image
Abda = image.load_img('original images/AbdA_00013_m_31_i_fr_nc_sr_2016_2_e0_nl_o.jpg',target_size=(64,64))
Abdh = image.load_img('original images/AbdH_00129_m_24_o_nf_nc_hp_2016_1_e0_nl_o.jpg',target_size=(64,64))



# Testing

image = image.load_img('AlaG_00368_m_32_i_nf_nc_no_2016_2_e0_nl_o 7.jpg',target_size=(64,64))
test_image =  img_to_array(image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
index = training_set.class_indices

def prediction(result):
    if result[0][0]>=0.5:
        predict = 'Dog'
    else:
        predict = 'Cat'
    print('Prediction')
    return predict

prediction(result)