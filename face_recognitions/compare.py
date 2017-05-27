#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/26-11:14
description : 对比两张照片
"""
import cv2
import numpy
import argparse

import os
import sys
sys.path.append('../LBP')
import distance

import dlib

import api

import Face

parser = argparse.ArgumentParser()
parser.add_argument('trainFolder', help='Please provide the train folder name.')
parser.add_argument('testImagePath', help="Please provide the test image path.")
parser.add_argument('personNumber', help="Please provide the person number you want to train!")
parser.add_argument('distanceType', help="Please provide the method name.")
args = parser.parse_args()

if args.distanceType == 'L1':
	distanceMethod = distance.L1
elif args.distanceType == 'Euler':
	distanceMethod = distance.Euler
elif args.distanceType == 'cosine':
	distanceMethod = distance.cosine
else:
	distanceMethod = None

Max_Image_Size = (500, 500)

# Face 类的实例
faceHandle = Face.Face(distance=distanceMethod)

# 开始训练图像
faceHandle.train(args.trainFolder, isContainFolder=True, isDebug=True, personNumber=int(args.personNumber), imagePerPerson=6)
#测试图像

testImage = cv2.imread(args.testImagePath)
result = faceHandle.predict(testImage)
print(result)


#result : [[rects], [scores], [labels]]
for r in result:
	rect = r[0]
	scores = r[1]
	labels = r[2]
	for image in labels:
		cv2.imshow(image, cv2.imread(image) )
	cv2.rectangle(testImage, rect[:2], rect[2:], (255, 0, 0), 2)
	cv2.imshow('testImage', testImage)
	cv2.waitKey(0) & 0XFF
cv2.destroyAllWindows()