#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
description : 使用我自己的LBP算法得到的最后结果与opencv的LBP差不多（真的差不多）
"""
import LBPrecognize as LBP
import cv2
import read_feret as RF
import numpy
import cPickle

import  sklearn_LBP as SL
lbp = SL.LocalBinaryPatterns(16, 2)

IMAGE_SIZE = (90, 90)

trainData, trainLabel, testData, testLabel = RF.getFacesData('FERET/')
# fileName = 'feretData.txt'
# fs = open( fileName, 'r' )
# data = cPickle.load( fs )
# fs.close()

data = {
    "trainData" : [lbp.describe(item.reshape(IMAGE_SIZE)) for item in trainData],
    'testData' : [lbp.describe(item.reshape(IMAGE_SIZE)) for item in testData]
}

print "Starting split image into 16 block---------"
# 5效果比较好的
trainLBPData = numpy.array([LBP.splitBlock(item.reshape(IMAGE_SIZE), 5, 5)
                            for item in data['trainData']]).astype('float32')
testLBPData = numpy.array([LBP.splitBlock(item.reshape(IMAGE_SIZE), 5, 5)
                           for item in data['testData']]).astype('float32')

print "Spliting is done, starting prediction--------------"

svm_params = dict(kernel_type = cv2.SVM_LINEAR,
                   svm_type = cv2.SVM_C_SVC,
                   C=2.67,
                   gamma=5.383)
svm = cv2.SVM()

svm.train(trainLBPData, trainLabel.ravel(), params=svm_params)

correctCount = 0
nearCount = 0
#
# for index, testImage in enumerate(testLBPData):
#     predict = model.predict(testImage)[0]

# for index, testImage in enumerate(testLBPData):
#     predict, allNear = LBP.recognize(trainLBPData, testImage)
#
#     # 通过最邻近来计算最终的预测值
#     nearPredict = LBP.filter(trainLabel, allNear)
#
#     # print nearPredict, testLabel[index], trainLabel[predict]
#
#     # predict, indexs = LBP.recognize(trainLBPData[:240], testImage)
#     print "the predict label and test label is: ", trainLabel[predict][0], testLabel[index][0], nearPredict
#
#     if trainLabel[predict][0] == testLabel[index][0]:
#         correctCount += 1
#     if nearPredict == testLabel[index][0]:
#         nearCount += 1


#     cv2.imshow('test LBP image', data['testData'][index].reshape(IMAGE_SIZE))
#     cv2.imshow("test origin image", testData[index].reshape(IMAGE_SIZE))
#     cv2.imshow('predict LBP image', data['trainData'][predict].reshape(IMAGE_SIZE))
#     cv2.imshow("predict origin image", trainData[predict].reshape(IMAGE_SIZE))
#     cv2.waitKey(0) & 0XFF
# cv2.destroyAllWindows()
print "The accuracy of correctly recognize is %d,%d"%(correctCount, len(trainData))