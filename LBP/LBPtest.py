#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import numpy
import cv2
import sys
sys.path.append('../readFaceDataset')
import read_feret as RF
#引入LBP实现
import sklearn_LBP as SLBP
import LBPClass as CLBP
import distance

trainData, trainLabel, testData, testLabel = RF.getFacesData('../FERET', number=70, isDebug=True)

def nomarlize(trainData, testData, isDebug=False):
    lbp = SLBP.LocalBinaryPatterns(16, 2)

    allTrainHists = None
    allTestHists = None
    count = 1
    #逐一获取每一张图片的lbp图片，然后计算直方图数据
    for image in trainData:
        if isDebug:
            print("calc the train %d histgram" % (count))
            count += 1
        img = lbp.descript(image)
        hist = distance.splitBlock(img)
        if allTrainHists is None:
            allTrainHists = numpy.array(hist).astype('float32')
        else:
            allTrainHists = numpy.vstack((allTrainHists, hist))

    count = 0
    #逐一获取每一张图片的lbp图片，然后计算直方图数据
    for image in testData:
        if isDebug:
            print("calc the test %d histgram" % (count))
            count += 1
        img = lbp.descript(image)
        hist = distance.splitBlock(img)
        if allTestHists is None:
            allTestHists = numpy.array(hist).astype('float32')
        else:
            allTestHists = numpy.vstack((allTestHists, hist))

    #nomalize the data, maybe we do not need it
    allTrainHists = distance.nomalize(allTrainHists)
    allTestHists = distance.nomalize(allTestHists)

    return allTrainHists, allTestHists

allTrainHists, allTestHists = nomarlize(trainData, testData, True)

# #实例化类
histCompare = CLBP.HistCompare(allTrainHists)
correctCount = 0

#现在获取图片以及将其归一化已经完事儿
for index, hist in enumerate(allTestHists):
    predict = histCompare.recognize(hist, distance.L1)
    if testLabel[index] == trainLabel[predict[0]] or testLabel[index] == trainLabel[predict[1]]:
        correctCount += 1
    # cv2.imshow('image', numpy.hstack([testData[index], trainData[predict[0]], trainData[predict[1]]]))
    # # cv2.imshow('img', hist)
    # cv2.waitKey(200) & 0XFF

# cv2.destroyAllWindows()

print("The correct accuracy is %d/%d:%f" % (correctCount, len(testLabel), float(correctCount / len(testLabel))))