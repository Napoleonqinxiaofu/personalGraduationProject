#_*_coding:utf-8_*_
########
#author:xiaofu.qin
#description:使用svm预测yalefaces的LBP图片,结果是识别效果很差，又因为我对svm不熟悉，所以就不准备用下去了，直接使用距离判断的方式，这样多好
########
from sklearn import svm
from sklearn.externals import joblib
import numpy
import cv2

import distance
import sklearn_LBP as SLBP

import sys
sys.path.append('../readFaceDataset')
import readDatabase as RD

trainData, trainLabel, testData, testLabel = RD.getFacesData('../yalefaces/yalefaces/', 'sad', True)

model = joblib.load('yaleface_svm_model.pkl')

lbp = SLBP.LocalBinaryPatterns(16, 2)

for index, image in enumerate(testData):
	lbpImg = lbp.descript(image)
	hist = distance.splitBlock(lbpImg)

	predict = model.predict(hist)
	print(predict)