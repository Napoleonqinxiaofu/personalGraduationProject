#_*_coding:utf-8_*_
########
#author:xiaofu.qin
#description:比较LBP方法在不同的光照条件下是否具有不变性，其实对光照的要求还是挺高的
########
import sklearn_LBP as SLBP
import LBPClass as CLBP
import os
import cv2
import numpy
from PIL import Image

def readImages(folder, lbpHandler, num=5, IMAGE_SIZE=(150, 150)):
	allImages = []
	allLbpImages = []
	count = 0

	for image in os.listdir(folder):
		if not image.endswith('pgm'):
			continue

		count += 1
		if count > num:
			break

		img = numpy.array(Image.open(os.path.join(folder, image)).convert('L'))
		img = cv2.resize(img, IMAGE_SIZE)
		lbpImage = lbpHandler.descript(img)

		allImages.extend([img])
		allLbpImages.extend([lbpImage])

	return (allImages, allLbpImages)

if __name__ == '__main__':
	imageFolder = 'yaleB37'
	lbpHandle = SLBP.LocalBinaryPatterns(8, 1)
	# lbpHandle = CLBP.LBP(8, 1)
	allImages, allLbpImages = readImages(imageFolder, lbpHandle, 8)

	cv2.imshow('origin image', numpy.hstack(allImages))
	cv2.imshow('lbp image', numpy.hstack(allLbpImages))
	cv2.waitKey(0) & 0XFF
	cv2.destroyAllWindows()