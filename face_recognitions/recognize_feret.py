#_*_coding:utf-8_*_
import cv2
import numpy
import face_recognition as face
import sys
sys.path.append('../readFaceDataset/')
import read_feret as RF

import api

IMAGE_SIZE = (90, 90)
"""
train, trainLabel, test, testLabel = RF.getFacesData('../FERET/',125,  True)

train1 = []
test1 = []
trainEncodings = []
newTrLabel = []
testEncodings = []
newTeLabel = []

for index, item in enumerate(train):
    item = cv2.cvtColor(item.reshape(IMAGE_SIZE), cv2.COLOR_GRAY2BGR)
    try:
        #确保要有人脸
        face_locations = face.face_locations(item)
    except:
        face_locations = []

    if len(face_locations) < 1:
        print("no boxes in image")
        continue

    try:
        encode = face.face_encodings(item)[0]
    except Exception as err:
        print(err)
        continue
    print("train %d" % (index))
    trainEncodings.extend([encode])
    newTrLabel.append(trainLabel[index])
    train1.extend([item])

train = None
for index, item in enumerate(test):
    item = cv2.cvtColor(item.reshape(IMAGE_SIZE), cv2.COLOR_GRAY2BGR)
    try:
        #确保要有人脸
        face_locations = face.face_locations(item)
    except:
        face_locations = []

    if len(face_locations) < 1:
        print('No boxes in the test image')
        continue

    try:
        encode = face.face_encodings(item)[0]
    except:
        continue
    print("test %d" % (index))
    testEncodings.extend([encode])
    newTeLabel.append(testLabel[index])
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

RF.save('feretResNetData.txt', data)
"""

data = RF.load('feretResNetData.txt')
trainEncodings = data["trainEncodingData"]
train1 = data["originTrainData"]
newTrLabel = data["trainLabel"]
testEncodings = data["testEncodingData"]
test1 = data["originTestData"]
newTeLabel = data["testLabel"]


for index, item in enumerate(testEncodings):
    results = api.find_similar_face(trainEncodings, item)
    print(results)
    result = [newTrLabel[idx][0] for idx in results]
    print(result)
    print("This time label is %d" % (newTeLabel[index]))
    # cv2.imshow('test image', test1[index])
    # cv2.imshow('predict 1st image', train1[results[0]])
    # cv2.imshow('predict 2nd image', train1[results[1]])
    # cv2.imshow('predict 3rd image', train1[results[2]])
#     cv2.imshow('predict images', numpy.hstack([
#         train1[results[0]], train1[results[1]], train1[results[2]], test1[index]
#     ]))
#     cv2.waitKey(400) & 0XFF
# cv2.destroyAllWindows()