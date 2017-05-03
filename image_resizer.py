from PIL import Image
import os, sys

path = "/Users/administrator/PDFS/MachineLearning/december/movie trailer/data/validation/drama/"
path2 = "/Users/administrator/PDFS/MachineLearning/december/movie trailer/data/validation/drama2/" 
dirs = os.listdir(path)
#print dirs

def resize():
	for item in dirs:
		if item != '.DS_Store':
			if os.path.isfile(path+item):
				im = Image.open(path+item)
				f,e = os.path.splitext(path+item)

				imResize = im.resize((150,150), Image.ANTIALIAS)
				imResize.save(path2+item, 'JPEG', quality=90)

resize()