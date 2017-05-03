
import os
import sys
import subprocess
import cv2
import numpy as np 
from glob import glob


trainVideoFolder = '/home/khushi/Documents/midterm_movie_trailer/data_final/test'
trainFrameFolder = '/home/khushi/Documents/midterm_movie_trailer/data_final_frame/test'

def read_frames(videopath, genre, start_time=5000, end_time=120000, time_step=3000):
	
	print "Getting frame for", videopath
	try:
		cap = cv2.VideoCapture(videopath)
		for k in range(start_time, end_time+1, time_step):
			cap.set(cv2.CAP_PROP_POS_MSEC, k)
			succes, frame = cap.read()
			if succes:
				frame = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_AREA)
				yield frame
	except Exception as e:
		print e 
		return





def read_video(trainVideoFolder):
	dataSplit = os.listdir(trainVideoFolder)
	for genre in dataSplit:
		gFolder = os.path.join(trainVideoFolder, genre)
		if not os.path.exists(os.path.join(trainFrameFolder, genre)):
			os.makedirs(os.path.join(trainFrameFolder, genre))
		gList = os.listdir(gFolder)
		# print gFolder
		
		for i in range(len(gList)): 
			videopath = os.path.join(gFolder, gList[i])
			for frame_no, frame in enumerate(read_frames(videopath, genre)):
				frameWritepath=os.path.join(trainFrameFolder, genre, str(i)+'-'+str(frame_no)+'.jpg' )
				cv2.imwrite(frameWritepath, frame)







if not os.path.exists(trainFrameFolder):
	os.makedirs(trainFrameFolder)


read_video(trainVideoFolder)
