#new dataset and sgd optimizer
import os, sys
import theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, optimizer=fast_compile, exception_verbosity=high, floatX=float32 python Application.py"
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
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

# dimensions of our images.
img_width, img_height = 150, 150
seed = 7
np.random.seed(seed)

top_model_weights_path = 'bottleneck_fc_model_1.h5'
# train_data_dir = '/Users/administrator/PDFS/MachineLearning/december/movie trailer/data/train'
# validation_data_dir = '/Users/administrator/PDFS/MachineLearning/december/movie trailer/data/validation'
train_data_dir = '/home/khushi/Documents/midterm_movie_trailer/data_final_frame/train'
validation_data_dir = '/home/khushi/Documents/midterm_movie_trailer/data_final_frame/test'

nb_train_samples = 4134 + 3496 + 3412 + 3385
nb_validation_samples = 470 + 272 + 342 + 376
epochs = 100
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples, verbose=1)
    np.save(open('bottleneck_features_train_full.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples, verbose=1)
    np.save(open('bottleneck_features_validation_full.npy', 'w'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train_full.npy'))
    train_labels = np.array([0] * 4134 + [1] * 3496 + [2]*3412 + [3]*3385)
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_Y = encoder.transform(train_labels)
    Y_train = to_categorical(encoded_Y)



    validation_data = np.load(open('bottleneck_features_validation_full.npy'))
    validation_labels = np.array(
        [0] * 470 + [1] * 272 + [2]*342 + [3]*376)

    encoder.fit(validation_labels)
    encoded_y = encoder.transform(validation_labels)
    Y_val = to_categorical(encoded_y)


    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.summary()
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()


    history = model.fit(train_data, Y_train, epochs=epochs, batch_size=batch_size, verbose = 2, validation_data=(validation_data, Y_val)
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
    plt.savefig('accuracy_botleneck.jpg')
    # plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_bottleneck.jpg')
    # plt.show()


#save_bottlebeck_features()
train_top_model()
