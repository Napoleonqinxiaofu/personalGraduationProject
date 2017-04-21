#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
40张之中有一张预测出现了错误
"""
import cPickle
import numpy
import cv2
import LBPrecognize as LBP
import read_attr as RA

fileName = 'attFaceLBP_sklearn.txt'
fs = open(fileName, 'r')
data = cPickle.load(fs)
fs.close()
#"trainData" : trainData,
# "trainLabel" : trainLabel,
# "testData" : testData,
# "testLabel" : testLabel

print "Starting read the database from disk----------"
trainData, trainLabel, testData, testLabel = RA.getFacesData('./att_faces/')

print "Reading all image from disk is done----------"
allHist = [LBP.splitBlock(item, 5, 5) for item in data['trainData']]
testHist = [LBP.splitBlock(item, 5, 5) for item in data['testData']]

print "split all image into 25 small area is done-----------"
count = 0
for index, item in enumerate(testHist):
    predict, allNear = LBP.recognize(allHist, item)

    #通过最邻近来计算最终的预测值
    nearPredict = LBP.filter(trainLabel, allNear)

    if testLabel[index] == trainLabel[predict]:
        count += 1

    # print trainL[allNear[0]], trainL[allNear[1]], trainL[allNear[2]], \
    #     trainL[allNear[3]], trainL[allNear[4]], testL[index]
    cv2.imshow('predict image', data['trainData'][predict])
    cv2.imshow('test image', data['testData'][index])
    cv2.imshow('predict original image 1', trainData[predict].reshape((90, 90)))
    cv2.imshow('test original image 1', testData[index].reshape((90, 90)))
    cv2.waitKey(400) & 0XFF
cv2.destroyAllWindows()

print count