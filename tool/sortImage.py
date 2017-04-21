#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import os
import cv2
import numpy
import cPickle

import detect
import process

folder = u'E:/女明星人脸数据库/'
# for fileName in os.listdir( folder ):
#     # print fileName
#     # print os.path.isdir( folder +  fileName )
#     if not os.path.isdir(folder + fileName):
#         continue
#     count = 0
#     for innerFile in os.listdir( folder + fileName ):
#         if not innerFile.endswith( 'jpg' ):
#             os.remove( folder + fileName + '/' + innerFile )
#         else:
#             os.rename( folder+fileName+'/'+innerFile, folder+fileName+'/'+str(count)+'.jpg' )
#             count += 1

data = []

draw = detect.Draw()

def getMyData(folder='myData/'):
    data = []
    labels = []

    for fileName in os.listdir(folder):
        for innerFile in os.listdir(folder + fileName):
            face_rects = draw.detect(folder + fileName + '/' + innerFile)
            if face_rects is None:
                continue
            #读取灰度图片
            img = cv2.imread(folder + fileName + '/' + innerFile, 0)
            x, y, w, h = face_rects[0]

            data.append(cv2.resize(img[y:y+h, x:x+w], (60, 60)).flatten())
            labels.append(int(fileName))
    #每一种类别的第一一张图片作为测试图片
    trainData = None
    testData = None
    trainLabel = None
    testLabel = None

    count = -1
    for index, label in enumerate(labels):
        if label != count:
            count = label
            if testData is None:
                testData = data[index]
                testLabel = numpy.array([label])
            else:
                testData = numpy.vstack((testData, data[index]))
                testLabel = numpy.vstack((testLabel, [label]))
        else:
            if trainData is None:
                trainData = data[index]
                trainLabel = numpy.array([label])
            else:
                trainData = numpy.vstack((trainData, data[index]))
                trainLabel = numpy.vstack((trainLabel, [label]))

    return (trainData, trainLabel, testData, testLabel)


if __name__ == '__main__':
    import eigenfaces as PCA
    from skimage import feature
    import LBPClass as LBP

    IMAGE_SIZE = (60, 60)

    print "Starting get the image----------"

    trainData, trainLabel, testData, testLabel = getMyData()
    # mylbp = LBP.LBP()
    # trainData = [mylbp.descript(image.reshape(IMAGE_SIZE)) for image in trainData]
    # testData = [mylbp.descript(image.reshape(IMAGE_SIZE)) for image in testData]
    #
    # data = {
    #     "trainData": trainData,
    #     "trainLabel": trainLabel,
    #     "testData": testData,
    #     "testLabel": testLabel
    # }
    #
    # fs = open('beautyLBPDataSet.txt', 'w')
    # cPickle.dump(data, fs)
    # fs.close()
    # print "Saving the LBP data is done-----------"
    fs = open('beautyLBPDataSet.txt', 'r')
    data = cPickle.load(fs)
    fs.close()

    trainLBPData = data['trainData']
    # trainLabel = data['trainLabel']
    testLBPData = data['testData']
    # testLabel = data['testLabel']
    #
    # print "Starting calclate the LBP feature---------------"
    #
    # # trainData = [cv2.equalizeHist(image.reshape(IMAGE_SIZE)).flatten() for image in trainData]
    # # testData = [cv2.equalizeHist(image.reshape(IMAGE_SIZE)).flatten() for image in testData]
    #
    # trainLBPData = [feature.local_binary_pattern(image=image.reshape(IMAGE_SIZE), P=24,
    #                                        R=3, method='uniform').astype('uint8') for image in trainData]
    #
    # testLBPData = [feature.local_binary_pattern(image=image.reshape(IMAGE_SIZE), P=24,
    #                                           R=3, method='uniform').astype('uint8') for image in testData]
    #
    print "calculate LBP is done---------"
    lbp = LBP.LBPFace(trainLBPData)
    #
    # # print "Saving the images----------"
    # #
    # # print "Calculate the PCA matrix-----------"
    # #
    # # pca = PCA.PCAFace(trainData)
    # #
    # # IMAGE_SIZE = (60, 60)
    # #
    # # for index, image in enumerate(testData):
    # #     predict = pca.recognize(image)
    # #     cv2.imwrite(str(index)+'.jpg',
    # #                 numpy.hstack((image.reshape(IMAGE_SIZE), trainData[predict].reshape(IMAGE_SIZE))))
    # #     cv2.imshow('test image', image.reshape(IMAGE_SIZE))
    # #     cv2.imshow('predict image', trainData[predict].reshape(IMAGE_SIZE))
    # #     cv2.waitKey(0) & 0XFF
    # # cv2.destroyAllWindows()
    #
    #
    for index, image in enumerate(testLBPData):
        # lbp = feature.local_binary_pattern(image=image.reshape(IMAGE_SIZE), P=24,
        #                                    R=3, method='uniform')
        predict = lbp.recognize(image)
        cv2.imshow('test image', image)
        cv2.imshow('test image1', testData[index].reshape(IMAGE_SIZE))
        cv2.imshow('predict image', trainLBPData[predict])
        cv2.imshow('predict image1', trainData[predict].reshape(IMAGE_SIZE))
        # cv2.imwrite(str(index) + '.jpg',
        #             numpy.hstack((testData[index].reshape(IMAGE_SIZE),
        #                           trainData[predict].reshape(IMAGE_SIZE))))
        cv2.waitKey(0) & 0XFF
    cv2.destroyAllWindows()