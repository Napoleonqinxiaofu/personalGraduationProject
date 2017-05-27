#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import numpy
import cv2
from PIL import Image
import os
import pickle



def getFacesData(folder, number=50, isDebug=False, IMAGE_SIZE = (90, 90)):
    """
    从feret数据库提取人脸数据，每一个人的第10张图片作为测试图片
    :param folder: att_faces人脸数据库的路径
    :return: trainData， trainLabel， testData， testLabel
    """
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []
    count = 0

    for personFolder in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, personFolder)):
            continue
        count += 1
        if count > number:
            break
        #当前人脸数据的标签
        currentLabel = int(personFolder.split('-')[1])
        if isDebug:
            print("Current label is %d" % (currentLabel))

        for concrateImage in os.listdir(os.path.join(folder, personFolder)):
            if not concrateImage.endswith('tif'):
                continue

            imgPath = os.path.join(folder, personFolder, concrateImage)
            img = Image.open(imgPath).convert('L')
            img = numpy.array(img).astype('uint8')
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.equalizeHist(img)


            currentImage = int(concrateImage.split('.')[0])

            #使用第7张图像来作为测试图像
            if currentImage != 7:
                trainData.extend([img])
                trainLabel.append(currentLabel)
            else:
                testData.extend([img])
                testLabel.append(currentLabel)

    return trainData, trainLabel, testData, testLabel

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
    import LBPrecognize as LBP
    import cPickle
    import eigenfaces


    print("Starting to read the image from disk-------")
    trainData, trainLabel, testData, testLabel = getFacesData('../FERET/')
    print("Reading all image is done------------")
    print(trainData.shape)
    print("Starting calculate the lowDData of trainData-----")
    pcaFace = eigenfaces.PCAFace(testData)

    print("Starting prediction-------------")
    # count = 0
    # for index in numpy.arange(0, 200, 6):
    #     image = trainData[index]
    #     predict = pcaFace.recognize(image)
    #     print(trainLabel[predict], testLabel[index])
    #     if trainLabel[predict] == testLabel[index]:
    #         count += 1
    #     cv2.imshow('predict image', testData[predict].reshape(IMAGE_SIZE))
    #     cv2.imshow('test image', image.reshape(IMAGE_SIZE))
    #     cv2.waitKey(1000) & 0XFF
    # cv2.destroyAllWindows()
    # print(count, '/', testData.shape)

    #
    # print "Starting extract the LBP feature-------"
    # trainData = [(LBP.LBP(item.reshape(IMAGE_SIZE), 16, 2)).ravel() for item in trainData]
    # testData = [(LBP.LBP(item.reshape(IMAGE_SIZE), 16, 2)).ravel() for item in testData]
    #
    # print "Extracting the LBP feature is done----------"
    # data = {
    #     "trainData" : trainData,
    #     "trainLabel" : trainLabel,
    #     "testData" : testData,
    #     "testLabel" : testLabel
    # }
    # fileName = 'feretData.txt'
    # # fs = open(fileName, 'w')
    # # cPickle.dump(data, fs)
    # # fs.close()
    # fs = open( fileName, 'r' )
    # data = cPickle.load( fs )
    # fs.close()
    # for i in range(10):
    #     cv2.imshow('LBP', data['trainData'][i].reshape(IMAGE_SIZE))
    #     cv2.waitKey(0) & 0XFF
    # cv2.destroyAllWindows()