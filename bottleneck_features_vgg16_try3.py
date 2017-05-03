#new dataset and sgd optimizer
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import os, random
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten, Layer
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam, SGD
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 150, 150
seed = 7
np.random.seed(seed)

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/Users/administrator/PDFS/MachineLearning/december/movie trailer/data/train'
validation_data_dir = '/Users/administrator/PDFS/MachineLearning/december/movie trailer/data/validation'
nb_train_samples = 3134
nb_validation_samples = 1356
epochs = 50
batch_size = 32

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    learning_rate = 0.1
    # decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=0.001, momentum=momentum, decay=1e-2, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()


    history = model.fit(train_data, train_labels, nb_epoch=epochs, batch_size=batch_size, verbose = 2, validation_data=(validation_data, validation_labels)
        # callbacks = [
        #   keras.callbacks.ModelCheckpoint(top_model_weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
     #      keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    )
    model.save_weights(top_model_weights_path)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


save_bottlebeck_features()
train_top_model()
