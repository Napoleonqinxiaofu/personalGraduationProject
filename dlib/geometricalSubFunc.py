#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/23-9:25
description :  为geometricalFeaturesRecognize.py写一些辅助函数的
"""

import math

"""

dlib检测到的landmark关键点的分布如下:
			{
				IdxRange jaw;       // [0 , 16]
				IdxRange rightBrow; // [17, 21]
				IdxRange leftBrow;  // [22, 26]
				IdxRange nose;      // [27, 35]
				IdxRange rightEye;  // [36, 41]
				IdxRange leftEye;   // [42, 47]
				IdxRange mouth;     // [48, 59]
				IdxRange mouth2;    // [60, 67]
			}
			"""

#计算关键点中的两个人眼的中心位置，返回的顺序是——左眼到右眼
def findEyeCenterPointer(shape):
	points = []
	centers = []
	#先计算右眼睛的位置
	for i in range(42, 48):
		points.append((shape.part(i).x, shape.part(i).y))

	#根据x坐标从小到大进行排序
	points.sort(key=lambda x : x[0])

	centerx = int(points[0][0] + points[-1][0]) // 2
	centery = int(points[0][1] + points[-1][1]) // 2

	centers.append((centerx, centery))
	points = []

	# 计算左眼睛的位置
	for i in range(36, 42):
		points.append((shape.part(i).x, shape.part(i).y))

	# 根据x坐标从小到大进行排序
	points.sort(key=lambda x: x[1])

	centerx = int(points[0][0] + points[-1][0]) // 2
	centery = int(points[0][1] + points[-1][1]) // 2

	centers.append((centerx, centery))

	return centers

#计算鼻尖的坐标
def findNose(shape):
	return [(shape.part(30).x, shape.part(30).y)]

#计算嘴角的两个坐标,返回的顺序是第一个是左边的嘴角坐标，第二个是右边的嘴角坐标
def findMouth(shape):
	points = []
	centers = []
	# 先计算右眼睛的位置
	for i in range(60, 68):
		points.append((shape.part(i).x, shape.part(i).y))

	# 根据x坐标从小到大进行排序
	points.sort(key=lambda x: x[0])

	centerx = int(points[0][0] + points[-1][0]) // 2
	centery = int(points[0][1] + points[-1][1]) // 2

	centers.append(points[0])
	centers.append(points[-1])
	points = []

	return centers

# 计算两个坐标之间的距离
def distance(point1, point2):
	return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)