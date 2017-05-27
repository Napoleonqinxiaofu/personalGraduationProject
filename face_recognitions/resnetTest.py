#_*_coding:utf-8_*_
########
#author : xiaofu.qin
#description : 测试dlib的resnet深度残差网络的人脸识别在yalefaces、at_faces、feret数据库上的识别效果，没开始之前我就预测一定是100%。
########
import Face
import sys
sys.path.append('../readFaceDataset')
import getFacesData as FD 
sys.path.append('../LBP')
import distance
import numpy
import cv2

import argparse
parser = argparse.ArgumentParser()
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


trainData, trainLabel, testData, testLabel = FD.getFacesData('yalefaces', '../yalefaces/yalefaces', isDebug=True, isGray=False, imageNumber=200)

resnetFace = Face.Face(distance=distanceMethod)

resnetFace.train1(trainData, trainLabel, True)

wrongTime = 0

for index, image in enumerate(testData):
	result = resnetFace.predict(image)
	if len(result) == 0:
		wrongTime += 1
		continue
	#for s in result:
# 		cv2.imshow('allImage', numpy.hstack([
# 			image, trainData[s[3][0]], trainData[s[3][1]], trainData[s[3][2]]
# 		]))
# 		cv2.rectangle(image, (s[0][0], s[0][1]), (s[0][2], s[0][3]), 0, 2)
# 		cv2.imshow('img', image)
# 		cv2.waitKey(200) & 0xff
# cv2.destroyAllWindows()

print('The accuracy is %d/%d : %f' % (wrongTime, len(testData), float(wrongTime) /len(testData) ))