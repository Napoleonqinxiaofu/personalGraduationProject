#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/15-9:10
description :  人脸预处理的文件
"""
import cv2
import numpy

import detect
import faceAlign
import imtool
import process

"""
preTreatment 类的介绍
		在具体进行人脸识别之前做一下图像的预处理，预处理的方法有直方图均衡化和做一个椭圆形的掩膜，将其他部分去掉。首先将图片的尺寸归一化为targetImgSize，这个参数可以在初始化类的时候传递，也可以不传递，因为有默认值(50, 50)。提供的对外接口有main和getImg两个函数，main函数则表示进行所有的预处理步骤，getImg获取预处理之后的图像。

		类初始化参数以及开放函数的参数如下：
		类初始化：
				targetImgSize ： 元组， 表示程序会将图片缩放置该参数的值的大小，默认是(50, 50)
				axes ： 元组， 由于掩膜图像是一个椭圆，所以如果你将图像缩放成不同的分辨率的情况下，椭圆需要随之改变，我并没有通过程序来计算这个椭圆应该怎么变化，所以需要你主动传递这个参数，元组的第一个值表示横向的半长轴长度，第二个值表示纵向的半长轴长度。默认是(25, 30)

		main 函数：
			img ： 需要进行预处理的图片——numpy array

		getImg 函数：
			参数：无
			returns ： numpy array type ， img

"""

class PreTreatment:
	def __init__(self, targetImgSize=(50, 50), axes=(25, 30)):
		self.axes = axes
		self.img = None
		#将图像缩放的最终大小
		self.targetImgSize = targetImgSize

	def main(self, img):
		self.equalize(img)
		self.masking()

	def getImg(self):
		return self.img

	def _judegImageShape(self, img):
		# 判断当前传递进来的img是灰度图片还是彩色图片
		if len(img.shape) == 2:
			gray = img
		elif len(img.shape) == 3:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			gray = None
		return gray

	def equalize(self, img):
		# 对图像进行直方图均衡化
		img = self._judegImageShape(img)
		if img is None:
			raise Exception('请输入图像')

		self.img = img
		self.resize()

		self.img = cv2.equalizeHist(self.img)

	def masking(self):
		img = numpy.zeros(self.targetImgSize).astype('uint8')
		#椭圆中心
		centerPoint = (img.shape[0] / 2, img.shape[1] / 2)
		#椭圆的坐标轴的长度
		axes = self.axes
		#整个椭圆的偏转角度
		rotateAngle = 0
		#椭圆的起始角度
		startAngle = 0
		#椭圆终止的角度
		endAngle = 360
		#颜色
		color = 1
		#线条粗细
		thickness = -1
		cv2.ellipse(img, centerPoint, axes, rotateAngle, startAngle, endAngle, color, thickness)

		self.img = cv2.bitwise_and(self.img, self.img, mask=img)

	def resize(self):
		self.img = cv2.resize(self.img, self.targetImgSize, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
	preTreat = PreTreatment()
	faceAlign = faceAlign.FaceAlign()

	import os
	for path in os.listdir('image'):
		img = cv2.imread('image/' + path)
		alignImg = faceAlign.align(img)
		if alignImg is None:
			continue
		preTreat.main(alignImg)
		newImage = preTreat.getImg()

		# process.imshow(newImage, 'asd')
		cv2.imshow('asd', newImage)
		print path
		cv2.waitKey(500) & 0XFF

	cv2.destroyAllWindows()