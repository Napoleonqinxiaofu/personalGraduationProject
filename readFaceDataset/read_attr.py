#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import os
from PIL import Image
import numpy
import cv2
import pickle


def getFacesData(folder, isDebug=False, IMAGE_SIZE = (90, 90)):
    """
    从att_faces数据空提取人脸数据，每一个人的第10张图片作为测试图片
    :param folder: att_faces人脸数据库的路径
    :return: trainData， trainLabel， testData， testLabel
    """
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []

    for personFolder in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, personFolder)):
            continue

        #当前人脸数据的标签
        currentLabel = int(personFolder.replace('s', ''))
        if isDebug:
            print("current label is:%s" % (currentLabel) )

        for concrateImage in os.listdir(os.path.join(folder, personFolder)):
            imgPath = os.path.join(folder, personFolder, concrateImage)
            img = Image.open(imgPath).convert('L')
            img = numpy.array(img).astype('uint8')
            img = cv2.resize(img, IMAGE_SIZE)

            currentImage = int(concrateImage.split('.')[0])

            if currentImage != 10:
                trainData.extend([img])
                trainLabel.append(currentLabel)
            else:
                testData.extend([img])
                testLabel.append(currentLabel)

    return trainData, trainLabel, testData, testLabel

def getFacesData1(folder, isDebug=False, isPCA=True):
    """
    从att_faces数据空提取人脸数据，每一个人的第10张图片作为测试图片
    :param folder: att_faces人脸数据库的路径
    :return: trainData， trainLabel， testData， testLabel
    """
    trainData = None
    trainLabel = None
    testData = None
    testLabel = None

    for personFolder in os.listdir(folder):
        if os.path.isdir(folder + personFolder):
            #当前人脸数据的标签
            currentLabel = int(personFolder.replace('s', ''))
            if isDebug:
                print("current label is:%s" % (currentLabel) )

            for concrateImage in os.listdir(folder + personFolder + '/'):
                imgPath = folder + personFolder + '/' + concrateImage
                img = Image.open(imgPath).convert('L')
                img = numpy.array(img).astype('uint8')
                img = cv2.resize(img, IMAGE_SIZE).ravel()

                currentImage = int(concrateImage.split('.')[0])

                if currentImage != 10:
                    if trainData is None:
                        trainData = img
                        trainLabel = numpy.array([currentLabel])
                    else:
                        trainData = numpy.vstack((trainData, img))
                        trainLabel = numpy.vstack((trainLabel, [currentLabel]))
                else:
                    if testData is None:
                        testData = img
                        testLabel = numpy.array([currentLabel])
                    else:
                        testData = numpy.vstack((testData, img))
                        testLabel = numpy.vstack((testLabel, [currentLabel]))

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
    trainData, trainLabel, testData, testLabel = getFacesData('./att_faces/')
    import faceAlign.faceAlign
    import preTreatment

    align = faceAlign.FaceAlign()
    treat = preTreatment.PreTreatment()
    count = 0
    for person in os.listdir('./att_faces/'):
        if os.path.isdir('./att_faces/' + person):
            for people in os.listdir('./att_faces/' + person):
                imgPath = './att_faces/' + person + '/' + people
                img = Image.open(imgPath).convert('L')
                img = numpy.array(img).astype('uint8')

                img1 = align.getFaceOnly(img)
                # img = align.align(img)
                if img1 is not None:
                    img = img1
                else:
                    count += 1

                treat.main(img)
                img = treat.getImg()

                cv2.imshow('full image', img)
                cv2.waitKey(400) & 0XFF
    cv2.destroyAllWindows()
    print(count)