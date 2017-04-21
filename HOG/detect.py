#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/20-15:18
description : 检测物体的区域
"""
import cv2
import numpy

datasetPath = 'dataset'
SAMPLES = 4

def path(cls, i):
	return "%s/%s/%d.jpg" % (datasetPath, cls, i+1)

def getFlannMatcher():
	return cv2.FlannBasedMatcher(dict(
		algorithm=1,
		trees=5
	), {})

def getBowExtractor(extract, flann):
	return cv2.BOWImgDescriptorExtractor(extract, flann)

def getExtracDetector():
	return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()

def extractSIFT(imgPath, extractor, detector):
	img = cv2.imread(imgPath, 0)
	return extractor.compute(img, detector.detect(img))[1]

def bowFeatures(img, extractorBow, detector):
	return extractorBow.compute(img, detector.detect(img))

def carDetector():
	pos, neg = 'cars_train', 'basketball'
	detect, extract = getExtracDetector()
	matcher = getFlannMatcher()

	print('Building BOWKeannsTrainer')

	bowKmeansTrainer = cv2.BOWKMeansTrainer(1000)
	extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

	print('Adding the deatures to trainer')

	for i in range(SAMPLES):
		print(i)
		bowKmeansTrainer.add(extractSIFT(path(pos, i), extract, detect))
		bowKmeansTrainer.add(extractSIFT(path(neg, i), extract, detect))

	#实际上在这个地方花费的时间要多一些
	voc = bowKmeansTrainer.cluster()
	extract_bow.setVocabulary(voc)

	#准备训练数据以及标签
	trainData, trainLabel = [], []
	for i in range(SAMPLES):
		print("train image" + path(pos, i))
		trainData.extend(bowFeatures(cv2.imread(path(pos, i), 0), extract_bow, detect))
		trainLabel.append(1)
		trainData.extend(bowFeatures(cv2.imread(path(neg, i), 0), extract_bow, detect))
		trainLabel.append(-1)

	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setGamma(0.5)
	svm.setC(30)
	svm.setKernel(cv2.ml.SVM_RBF)

	svm.train(numpy.array(trainData), cv2.ml.ROW_SAMPLE, numpy.array(trainLabel))

	return svm, extract_bow