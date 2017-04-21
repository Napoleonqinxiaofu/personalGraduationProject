#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/21-9:07
description : 测试一下分类器
"""
import cv2
import numpy

from detect import bowFeatures, carDetector
from pyramid import pyramid
from nonMaximum import nonMaxSuppressionFast as nms
from slideWindow import slideWindow as sw

def in_range(number, test, thresh=0.2):
	return abs(number-test) < thresh


def getPath(i):
	return "dataset/cars_test/%d.jpg" % (i+1)

def rename(i):
	return "dataset/result/%d.jpg" % (i+1)

svm, extractor = carDetector()
detector = cv2.xfeatures2d.SIFT_create()
w, h = 100, 40

for i in range(2):

	img = cv2.imread(getPath(i))

	reactangles = []

	counter = 1
	scaleFactor = 1.25
	scale = 1
	font = cv2.FONT_HERSHEY_SIMPLEX

	for resized in pyramid(img, scaleFactor):
		scale = float(img.shape[1]) / float(resized.shape[1])

		for x, y, roi in sw(resized, 20, (w, h)):
			if roi.shape[1] != w or roi.shape[0] != h:
				continue

			try:
				bf = bowFeatures(roi, extractor, detector)
				_, result = svm.predict(bf)
				a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
				print("Class %d, score: %f" % (result[0][0], res[0][0]))
				score = score[0][0]

				if result[0][0] == 1:
					if score < -1.0:
						rx, ry, rx2, ry2 = int(x*scale), int(y*scale), int((x+w)*scale), int((y+h)*scale)
						reactangles.append([rx, ry, rx2, ry2, abs(score)])
			except:
				pass
			counter += 1

	windows = numpy.array(reactangles)
	boxes = nms(windows, 0.25)

	for r in boxes:
		(x, y, x2, y2, score) = r
		cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 1)
		cv2.putText(img, "%f" % score, (int(x), int(y)), font, 1, (0, 255, 0))

	cv2.imwrite(rename(i), img)
	# cv2.imshow('img', img)
	# cv2.waitKey(0) & 0XFF
	# cv2.destroyAllWindows()