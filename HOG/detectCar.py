#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/20-9:43
description : 使用SVM来训练分类检测某一张图片中是否存在汽车
"""
import cv2
import numpy
from sklearn import svm as sksvm
from os.path import join


SAMPLES = 20
dataPath = 'dataset'

pos, neg = 'cars_train', 'basketball'

def path(cls, i):
	return "%s/%s/%d.jpg" % (dataPath, cls, i+1)

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

#匹配
flaan_params = dict(
	algorithm = 1,
	trees = 5
)
flaan = cv2.FlannBasedMatcher(flaan_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flaan)

def extract_SIFT(img):
	image = cv2.imread(img, 0)
	return extract.compute(image, detect.detect(image))[1]


for i in range(SAMPLES):
	bow_kmeans_trainer.add(extract_SIFT(path(pos, i)))
	bow_kmeans_trainer.add(extract_SIFT(path(neg, i)))

voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)

def bow_features(img):
	im = cv2.imread(img, 0)
	return extract_bow.compute(im, detect.detect(im))

trainData, trainLabel = [], []

for i in range(SAMPLES):
	trainData.extend(bow_features(path(pos, i)))
	trainLabel.append(1)
	trainData.extend(bow_features(path(neg, i)))
	trainLabel.append(-1)
print('Starting create the svm instance')
# svm = cv2.ml.SVM_create()
# svm.train(numpy.array(trainData), cv2.ml.ROW_SAMPLE, numpy.array(trainLabel))
svm = sksvm.SVC()
svm.fit(trainData, trainLabel)
print('Trainng data action is done')

def predict(img):
	feature = bow_features(img)
	p = svm.predict(feature)
	return p

pos = 'cars_test'
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(10, 20):
	posImg = cv2.imread(path(pos, i))
	# negImg = cv2.imread(path(neg, i))

	posPredict = predict(path(pos, i))
	# negPredict = predict(path(neg, i))

	if(posPredict[0] == 1.0):
		cv2.putText(posImg, 'This picture is detected car', (10, 30), font, 1, (0, 255, 0), 2)

	# if (negPredict[0] == 1.0):
	# 	cv2.putText(negImg, 'This picture is not detected car', (10, 30), font, 1, (0, 255, 0), 2)
	cv2.imshow('posImg', posImg)
	# cv2.imshow('negImg', negImg)
	cv2.waitKey(0) & 0XFF
cv2.destroyAllWindows()