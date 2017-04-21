#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/21-15:28
description : 
"""
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure
import cv2


def getHogImage(img):
	fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
						cells_per_block=(3, 3), visualise=True)
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

	return (hog_image, hog_image_rescaled)

def getPath(folder, i):
	return "dataset/%s/%d.jpg" % (folder, i+1)

if __name__ == '__main__':
	import cv2
	import numpy

	# for i in range(10):
	# 	img = cv2.imread(getPath('basketball', i), 0)
	# 	img1 = cv2.imread(getPath('basketball', i), 0)
	# 	# hogImage, hogImage2 = getHogImage(img1)
	# 	hog = cv2.HOGDescriptor()
	# 	h = hog.compute(img)
	# 	print(numpy.array(h).shape)

	# 	cv2.imshow('origin image', numpy.hstack([img]))
	# 	cv2.imshow('hog', hogImage)
	# 	cv2.waitKey(0) & 0xff
	# cv2.destroyAllWindows()