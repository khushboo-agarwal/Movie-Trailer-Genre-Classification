'''
This code reads the already extracted frames in number and also the tags. 
'''



import cv2
import numpy as np 
import os
import glob
from natsort import natsorted
def makeVideoSegments(outputlist, label, x, y):
	segment = []
	# vidSeg = []
	# print len(outputlist)
	for k in range(1, len(outputlist)):
		if k == 0:
			segment.append(outputlist[k])
		if (k%8 != 0):
			# print k
			segment.append(outputlist[k])
		elif (k%8 == 0):
			segment.append(outputlist[k])
			seg = np.asarray(segment)
			# print seg.shape
			seg = np.rollaxis(np.rollaxis(seg,2,0),2,0)
			# print seg.shape
			
			segment = []	
			x.append(seg)
			if label == 'action':
				y.append('action')
			if label == 'comedy':
				y.append('comedy')
			if label == 'drama':
				y.append('drama')
			if label == 'horror':
				y.append('horror')	
	
	return x, y



def read_frames_video(videopath, label, x, y):
	
	framelist = os.listdir(videopath)
	framelist = natsorted(framelist)
	outputlist = []
	# y_train = []
	
	# print framelist
	for frame in framelist:
		framepath = os.path.join(videopath, frame)
		outputlist.append(cv2.imread(framepath, cv2.IMREAD_COLOR))
		# y_train.append(label)
	x, y= makeVideoSegments(outputlist, label, x, y)
	
	# print len(outputlist), len(y_train)
	return x, y






def main(folder):
	y=[]
	x = []
	outputlist =[]
	genre = sorted(os.listdir(folder))
	# print genre ->  ['action', 'comedy', 'drama', 'horror']
	for label in genre:
		genrePath = os.path.join(folder, label)
		videoFolder = os.listdir(genrePath)
		returnList = []
		for vname in sorted(videoFolder):
			videopath = os.path.join(genrePath, vname)
			x, y = read_frames_video(videopath, label, x, y)
	
	# outputlist = np.asarray(outputlist)
	# print y_train
	return x, y



trainFrameFolder = '/home/khushi/Documents/data_final_frames/train'  #changed the folder for data frames of size 224x224 to 112x112
trainVideoFolder = '/home/khushi/Documents/data_final/train'

validationFrameFolder = '/home/khushi/Documents/data_final_frames/validation'
validationVideoFolder = '/home/khushi/Documents/data_final/validation'



genre = []
if __name__ == '__main__':
	outputlist = []
	y_train=[]
	x_train = []
	x_val = []
	y_val = []
	x_train, y_train =  main(trainFrameFolder)
	x_val, y_val = main(validationFrameFolder)

	# print "VidSeg", np.asarray(vidSeg).shape
	# print len(y_train)
	# makeVideoSegments(outputlist, y_train, )
	# print outputlist.shape
	# print ""
	# print len(y_train)

# lst = makeVideoSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,colorFlag=colorFlag, flow=flow))