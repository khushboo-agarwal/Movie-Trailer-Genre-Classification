from __future__ import division
import math
import cv2
import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt 
import sys, os
import random as rn 
import time
import scipy
import sklearn
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
train_data_dir = '/home/khushi/Documents/midterm_movie_trailer/data_final_frame/train'
validation_data_dir = '/home/khushi/Documents/midterm_movie_trailer/data_final_frame/test'

nb_train_samples = 4134 + 3496 + 3412 + 3385
nb_validation_samples = 470 + 272 + 342 + 376

label = []
image = []

def hog(gray):
	gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
    # quantizing binvalues in (0...16)
	bins = np.int32(bin_n*ang/(2*np.pi))
    
	# Divide to 4 sub-squares
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	return hist


def read_image(imagepath):
	I = Image.open(imagepath)
	I1 = np.array(I)
	gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	return gray



def read_folder(train_data_dir):
	image = []
	label = []
	directory = os.listdir(train_data_dir)
	for genre in directory:
		
		genrePath = os.path.join(train_data_dir, genre)
		im = os.listdir(genrePath)
		for i in im:
			imagePath = os.path.join(genrePath, i)
			gray = read_image(imagePath)
			image.append(hog(gray))
			label.append(genre)
	image = np.array(image)
	# label = np.array(label)
	return image, label

def Evaluation(train_data, train_label, test_data, test_label):
	clf = svm.SVC( kernel = 'linear', C = 1, gamma = 0.0000001, verbose=True)
	clf.fit(train_data, train_label)
	predicted = clf.predict(test_data)
	error = (test_label == predicted).mean()
	label = ['action', 'comedy', 'horror', 'drama']
	arr = confusion_matrix(label_test, predicted, label)
	print arr
	print 'accuracy is %.2f %%' %(error*100) 

bin_n = 16

train_data = []
train_label = []
test_data = []
test_label = []

train_data, train_label = read_folder(train_data_dir)
test_data, test_label = read_folder(validation_data_dir)

print train_data.shape
print len(train_label)
print test_data.shape
print len(test_label)

Evaluation(train_data, train_label, test_data, test_label)



