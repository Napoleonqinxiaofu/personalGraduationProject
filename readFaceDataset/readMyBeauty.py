#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
description : 读取女明星人脸数据库
"""

import os
import cv2
import pickle
import sys
sys.path.append('../tool')
import imtool

def getFacesData(folder, isDebug=False, width=200):
	trainData, trainLabel, testData, testLabel = [], [], [], []
	#进入所有的人脸的目录
	for image in os.listdir(folder):
		currentPersonLabel = int(image)
		#接下来进入某一个人脸的目录
		count = 0
		testCount = 0
		for item in os.listdir(os.path.join(folder, image)):
			count += 1
			img = cv2.imread(os.path.join(folder, image, item), 3).astype('uint8')
			img = imtool.resize(img, width=width)
			if isDebug:
				print("Processing %s" % (os.path.join(folder, image, item)))
			# 指定训练图像的数量为20张,也为测试数据多加上几个图片
			if count > 20:
				testData.extend([img])
				testLabel.append(currentPersonLabel)
				testCount += 1
				if testCount > 4:
					break
				continue

			trainData.extend([img])
			trainLabel.append(currentPersonLabel)

	return (trainData, trainLabel, testData, testLabel)

def save(fileName, data):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f)
        f.close()

def load(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

if __name__ == '__main__':
	import face_recognition as face
	import sys
	sys.path.append('../face_recognition/')
	import api
	"""
	train, trl, test, tel = getFacesData('../myData', True)
	train1 = []
	test1 = []
	trainEncodings = []
	newTrLabel = []
	testEncodings = []
	newTeLabel = []
	for index, item in enumerate(train):
		#确保要有人脸
		face_locations = face.face_locations(item)
		if len(face_locations) < 1:
			continue
		trainEncodings.extend([face.face_encodings(item)[0]])
		newTrLabel.append(trl[index])
		train1.extend([item])

	train = None
	for index, item in enumerate(test):
		#确保要有人脸
		face_locations = face.face_locations(item)
		if len(face_locations) < 1:
			continue
		testEncodings.extend([face.face_encodings(item)[0]])
		newTeLabel.append(tel[index])
		test1.extend([item])

	test = None

	data = {
		"trainEncodingData" : trainEncodings,
		"originTrainData" : train1,
		"trainLabel" : newTrLabel,
		"testEncodingData" : testEncodings,
		"originTestData" : test1,
		"testLabel" : newTeLabel
	}

	save('myBeautyResNetData.txt', data)

	data = None
	"""
	data = load('myBeautyResNetData.txt')
	trainEncodings = data["trainEncodingData"]
	train1 = data["originTrainData"]
	newTrLabel = data["trainLabel"]
	testEncodings = data["testEncodingData"]
	test1 = data["originTestData"]
	newTeLabel = data["testLabel"]


	for index, item in enumerate(testEncodings):
		results = api.find_similar_face(trainEncodings, item)
		print(results)
		result = [newTrLabel[idx] for idx in results]
		print(result)
		print("This time label is %d" % (newTeLabel[index]))
		cv2.imshow('test image', test1[index])
		cv2.imshow('predict 1st image', train1[results[0]])
		cv2.imshow('predict 2nd image', train1[results[1]])
		cv2.imshow('predict 3rd image', train1[results[2]])
		cv2.waitKey(0) & 0XFF
	cv2.destroyAllWindows()