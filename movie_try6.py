import os, sys
import theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, optimizer=fast_compile, exception_verbosity=high, floatX=float32 python Application.py"
from glob import glob
import cv2
import numpy as np
import math
# from model_utils import get_features_batch
# from utils import dump_pkl
import videoprocessing_final_1 as vp 
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import *
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.initializers import glorot_uniform
from keras.callbacks import LearningRateScheduler
from keras import regularizers




trainFrameFolder = '/home/khushi/Documents/data_final_frames/train'  #changed the folder for data frames of size 224x224 to 112x112
trainVideoFolder = '/home/khushi/Documents/data_final/train'

validationFrameFolder = '/home/khushi/Documents/data_final_frames/validation'
validationVideoFolder = '/home/khushi/Documents/data_final/validation'

x_train = []
y_train = []

x_val = []
y_val = []
seed = 7
np.random.seed(seed)

x_train, y_train = vp.main(trainFrameFolder)
x_val, y_val = vp.main(validationFrameFolder)


x_train = np.asarray(x_train)
x_val = np.asarray(x_val)

print 'x_train:', x_train.shape
print 'x_val:', x_val.shape

print 'length of y train',  len(y_train)
print 'length of y val',  len(y_val)

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

num_train_samples = len(x_train)
num_val_samples = len(x_val)
print 'num_train_samples:', num_train_samples
print 'num_val_samples:', num_val_samples


img_width, img_height = 112, 112
# nb_train_samples = 373
batch_size = 10
nb_epoch = 100
nb_classes = 4

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
Y_train = to_categorical(encoded_Y)


encoder.fit(y_val)
encoded_y = encoder.transform(y_val)
Y_val = to_categorical(encoded_y)



# datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization=True, width_shift_range = 0.2, horizontal_flip=True, height_shift_range=0.2, fill_mode='nearest')

# datagen.fit(x)
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


#defining the model

def get_model(summary=False):

	'''
	Returns a keras model of the network
	'''

	model = Sequential()

	#1st layer group
	model.add(Convolution3D(32,  3, 3, 3,  activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), kernel_initializer='glorot_uniform', input_shape=(112, 112, 8, 3)))
	model.add(BatchNormalization(axis=2))
	model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='valid', name='pool1'))

	#2nd layer group
	model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', use_bias=True, name='conv2', subsample=(1, 1, 1)))
	model.add(BatchNormalization(axis=2))
	model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), border_mode='valid', name='pool2'))

	#3rd layer group
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same',use_bias=True, name='conv3', subsample=(1, 1, 1)))
	model.add(BatchNormalization(axis=2))
	model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), border_mode='valid', name='pool3'))


	#3a layer group
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same',use_bias=True, name='conv3a', subsample=(1, 1, 1)))
	model.add(BatchNormalization(axis=2))
	model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), border_mode='valid', name='pool3a'))
	

	#4th layer group
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', use_bias=True, name='conv4a',subsample=(1, 1, 1)))
	model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1)))
	model.add(BatchNormalization(axis=2))
	model.add(ZeroPadding3D(padding=(1, 1, 0)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4'))

	# #5th layer group
	model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same',use_bias=True, name='conv5a', subsample=(1, 1, 1)))
	model.add(BatchNormalization(axis=2))
	# model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5b', subsample=(1, 1, 1)))
	model.add(MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), border_mode='valid', name='pool5'))

	# 6th layer group
	model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv6a', subsample=(1, 1, 1)))
	model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv6b', subsample=(1, 1, 1)))
	# model.add(ZeroPadding3D(padding=(1, 1, 0)))
	model.add(MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), border_mode='valid', name='pool6'))
	# model.add(Dropout(0.5))
	model.add(Flatten())

	#Fully Connected Layer Group
	model.add(Dense(4096, activation='relu', name = 'fc7'))
	model.add(Dropout(0.5))
	
	model.add(Dense(4096, activation='relu', name = 'fc8'))
	# model.add(BatchNormalization(axis=2))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', name = 'fc9'))
	# model.add(BatchNormalization(axis=2))
	model.add(Dropout(0.5))
	model.add(Dense(4, activation = 'softmax', name = 'fc10'))
	# model.add(BatchNormalization(axis=2))
	if summary:
		print(model.summary())
	return model

model = get_model(summary = True)

# model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-5, momentum=0.5), metrics=[ 'accuracy'])
sgd = SGD(lr=0.001,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=[ 'accuracy', metrics.categorical_accuracy])

#Data augmentation:

# train_datagen = ImageDataGenerator()

# print('training')

# train_generator = train_datagen.flow_from_directory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]


history = model.fit(x_train, Y_train, validation_data=(x_val, Y_val), epochs=20 , batch_size=batch_size, verbose=2, shuffle=True)
model.save_weights('trailer_after_first80.h5', overwrite=True)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# plt.savefig('accuracy_100_5_transparent.jpg', transparent=True)
# plt.save('accuracy_100_5.jpg')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.savefig('loss_ne_100_5_adam.jpg', transparent=True)
plt.savefig('loss_ne_100_5_adam_notrans.jpg')
