#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/23-9:12
description :  通过dlib来计算出人脸的眼睛的中心坐标、鼻尖、嘴角的两个坐标，通过计算几何长度的比例来对人脸进行识别
"""

import cv2
import numpy
import dlib

import geometricalSubFunc as helpFuncs
import faceAlign


class GeometricalFace:
	def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat'):
		#当前人脸的5个关键点存储列表
		self.landmark = None
		#当前五个关键点之间的距离之比的存储位置
		self.distance = None

		#landmark68个关键点的坐标存放变量
		self.shape = None

		#人脸扶正的实例
		self.faceAlign = faceAlign.FaceAlign()

		#检测人脸区域的
		self.detector = dlib.get_frontal_face_detector()
		#加载dlib已经训练好的人脸关键点的数据文件
		self.predictor = dlib.shape_predictor(predictor_path)

	#开始检测关键点，img参数是一个已经被读取进来的numpy多维数组，可以不是灰度图像
	def detect(self, img, isDebug=False):
		self.landmark = None
		self.shape = None

		#下面是dlib给出的解释
		# Ask the detector to find the bounding boxes of each face. The 1 in the
		# second argument indicates that we should upsample the image 1 time. This
		# will make everything bigger and allow us to detect more faces.
		dets = self.detector(img, 1)

		if isDebug:
			print("Number of faces detected: {}".format(len(dets)))

		for k, d in enumerate(dets):
			if isDebug:
				print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
				k, d.left(), d.top(), d.right(), d.bottom()))

			# 获取68个landmark关键点的位置
			self.shape = self.predictor(img, d)

			eyeCenters = helpFuncs.findEyeCenterPointer(self.shape)
			noseCenter = helpFuncs.findNose(self.shape)
			mouthCenters = helpFuncs.findMouth(self.shape)

			eyeCenters.extend(noseCenter)
			eyeCenters.extend(mouthCenters)

			#只允许有一个人，所以永远都是最后被检测到的人被保留下来
			self.landmark = eyeCenters
			eyeCenters = None
			noseCenter = None
			mouthCenters = None


	#计算右眼到鼻尖与鼻尖到右嘴角的距离之比以及左边的眼镜中心到鼻尖的距离与鼻尖到左嘴角的距离之比
	def calcRatio(self):
		#首先计算右眼到鼻尖与鼻尖到右嘴角的距离之比
		leftEyeToNose = helpFuncs.distance(self.landmark[0], self.landmark[2])
		leftMouthToNose = helpFuncs.distance(self.landmark[2], self.landmark[3])

		rightEyeToNose = helpFuncs.distance(self.landmark[1], self.landmark[2])
		rightMouthToNose = helpFuncs.distance(self.landmark[2], self.landmark[4])

		self.distance = [float(leftEyeToNose)/leftMouthToNose, float(rightEyeToNose)/rightMouthToNose, float(leftEyeToNose+rightMouthToNose) / (rightEyeToNose+leftMouthToNose)]

	#将人脸扶正，计算两个眼睛中心之间的角度，然后进行旋转即可
	def align(self, img):
		img = self.faceAlign.align(img, self.landmark[1], self.landmark[0])
		return img

	#根据landmark中所有的点的坐标来选取人脸区域，尽量减少头发之类的干扰
	def clipFace(self, img, size=(60, 60)):
		if self.shape is None:
			return cv2.resize(img, size)

		points = []
		for i in range(68):
			points.append([self.shape.part(i).x, self.shape.part(i).y])

		points.sort(key=lambda x : x[0])
		left = points[0][0] if points[0][0] > 0 else 0
		right = points[-1][0] if points[-1][0] > 0 else 0

		points.sort(key=lambda x: x[1])
		top = points[0][1] if points[0][1] > 0 else 0
		bottom = points[-1][1] if points[-1][1] > 0 else 0;

		# print(left, right, top, bottom)

		if len(img.shape) > 2:
			img = cv2.resize(img[top:bottom, left:right, :], size)
		elif len(img.shape) == 2:
			img = cv2.resize(img[top:bottom, left:right], size)

		return img

	def drawPoints(self, img):
		if len(self.landmark) < 1:
			return False
		for point in self.landmark:
			cv2.circle(img, point, 2, (255, 0, 0), -1)
		cv2.imshow('img', img)
		cv2.waitKey(1000) & 0XFF
		cv2.destroyAllWindows()


if __name__ == '__main__':
	import os
	import sys

	sys.path.append('../readFaceDataset/')
	import readMyBeauty as RA
	sys.path.append('../LBP/')
	import sklearn_LBP as sklearnLBP

	#读取attr上所有图片的lbp特征并保存起来
	trainData, trainLabel, testData, testLabel = RA.getFacesData('../myData/', True)

	geometrical = GeometricalFace()
	lbp = sklearnLBP.LocalBinaryPatterns(24, 3)

	folder = '../image'
	def path(image):
		return "%s/%s" % (folder, image)
	count = 0

	train = []
	test = []

	IMAGE_SIZE = (60, 60)
	for index, image in enumerate(trainData):
		# img = image.reshape((90, 90, 3))
		img = image
		print("extract trainData %d" % (index))
		# img = cv2.equalizeHist(img[:, :, 0].astype('uint8'))
		# print(img.shape)
		# continue

		geometrical.detect(img)

		if geometrical.shape is None:
			img1 = cv2.resize(img, IMAGE_SIZE)
			count += 1
		else:
			img1 = geometrical.align(img)
			geometrical.detect(img1)
			#如果这一次找不到人脸的区域，那么直接返回传递进去的人脸图像
			img1 = geometrical.clipFace(img1)

		#提取LBP特征
		img1 = lbp.describe(img1)

		train.extend([img1])

	for index, image in enumerate(testData):
		# img = image.reshape((90, 90))
		img = image
		print("extract testData %d" % (index))
		# img = cv2.equalizeHist(img)

		geometrical.detect(img)

		if geometrical.shape is None:
			img1 = cv2.resize(img, IMAGE_SIZE)
			count += 1
		else:
			img1 = geometrical.align(img)
			geometrical.detect(img1)
			#如果这一次找不到人脸的区域，那么直接返回传递进去的人脸图像
			img1 = geometrical.clipFace(img1)

		#提取LBP特征
		img1 = lbp.describe(img1)

		test.extend([img1])


	# print(count)
	#
	# print( trainData.shape, train[0].shape, testData.shape, len(test))
	print('Saving data into file')
	RA.save('myBeautyLBPData.txt', {
		"radius" : 3,
		"points" : 24,
		"trainData" : train,
		"originTrainData" : trainData,
		"trainLabel" : trainLabel,
		"testData" : test,
		"originTestData" : testData,
		"testLabel" : testLabel
	})

	trainData = None
	testData = None
	train = None
	test = None
	trainLabel = None
	testLabel = None

