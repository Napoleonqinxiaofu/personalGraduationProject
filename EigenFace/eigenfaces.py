#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
description : 基于PCA算法的人脸识别算法实现，其实这个脚本的代码实现的是PCA的另外一种版本，
             它并没有做出那些对特征向量进行排序之类的操作，而只是过滤了某一些特征值小于某一个
             阈值的特征向量，所以我们可以试一下看看能不能实现我在论文里看到的对特征向量进行排序
             的算法。对yalefaces的测试错了一个,不管是图像的尺寸为多少
"""
from PIL import Image
import cv2
import numpy
import os
import sys
import time


import readDatabase as RD

def createDataSet(path):
    print("Starting read the image from disk------------")
    trainData, trainLabel, testData, testLabel = RD.getFacesData('yalefaces/yalefaces/', 'sad')
    print("Reading all images done--------------")

    trainData = numpy.array([cv2.resize(item, IMAGE_SIZE).ravel() for item in trainData])
    testData = numpy.array([cv2.resize(item, IMAGE_SIZE).ravel() for item in testData])

    return (trainData, trainLabel, testData, testLabel)

def eigenfaceCore(imageDataset):
    #获取imageDataset中的图像数以及每一张图像的所有的像素点的个数
    imageNumber, perImagePixels = imageDataset.shape

    #按列计算imageDataseset的平均值,这是一个行向量
    meanMatrix = numpy.mean(imageDataset, axis=0)

    #计算imageDataset与meanMatrix之间的差值
    diffMatrix = imageDataset - meanMatrix

    #计算协方差矩阵C的替代L
    diffMatrix = numpy.mat(diffMatrix)
    #使用点乘的矩阵运算方式获得协方差矩阵
    L = diffMatrix * diffMatrix.T

    #计算特征值和特征向量
    eigValues, eigVectors = numpy.linalg.eig(L)

    # eigVectors = eigVectors.T

    newEigVectors = None

    # for i in range(0, imageNumber):
    #     if eigValues[i] >= 1:
    #         if newEigVectors is None:
    #             newEigVectors = eigVectors[i]
    #         else:
    #             newEigVectors = numpy.vstack((newEigVectors, eigVectors[i]))
    #不用过滤了
    # newEigVectors = eigVectors.T

    eigFaces = diffMatrix.T * eigVectors

    return eigFaces, diffMatrix.T, meanMatrix


def pcaFunc(imageDataset, eigenNumber=1000):
    """
    实现在论文中所看到的PCA算法
    :param imageDataset: 所有的训练图像的集合，每一张图片是一个行向量
            eigenNumber : 我们想要的获得的特征向量的个数
    :return: 特征变换矩阵
    """
    # 获取imageDataset中的图像数以及每一张图像的所有的像素点的个数
    baseTime = time.time()
    imageNumber, perImagePixels = imageDataset.shape

    # 按列计算imageDataseset的平均值,这是一个行向量
    meanMatrix = numpy.mean(imageDataset, axis=0)
    print(time.time()-baseTime, 'first')
    # 计算imageDataset与meanMatrix之间的差值
    diffMatrix = imageDataset - meanMatrix

    # 计算协方差矩阵C的替代L
    diffMatrix = numpy.mat(diffMatrix)
    # 使用点乘的矩阵运算方式获得协方差矩阵
    L = diffMatrix * diffMatrix.T
    print(time.time() - baseTime, 'second')

    # 计算特征值和特征向量
    eigValues, eigVectors = numpy.linalg.eig(L)
    print(time.time() - baseTime, 'third')
    eigVectors = eigVectors.T

    #numpy已经为所有的特征值进行排序，最大的在最前面
    eigFaces = diffMatrix.T * eigVectors[:eigenNumber].T
    print(time.time() - baseTime, 'fourth')
    return eigFaces, diffMatrix.T, meanMatrix

def recognize(testImage, diffMatrix, meanMatrix, eigFaces):
    """
    实现人脸识别的方法
    :param testImage: 需要进行识别的图像
    :param diffMatrix: 总体训练样本与平均向量之间差值的矩阵
    :param meanMatrix: 总体训练样本的平均矩阵
    :param eigFaces: 特征变换矩阵
    :return: corresponseIndex : 由计算出的最小距离的图像的标签
    """
    #获取特征空间的图片数目以及每张图片的像素的个数
    perImagePixels, imageNumber = eigFaces.shape

    #将每个样本投影到特征空间
    projectImage = eigFaces.T * diffMatrix

    diffTestImage = testImage - meanMatrix

    diffTestImage = numpy.mat(diffTestImage)

    #将测试的图片投影到特征空间里
    projectTestImage = eigFaces.T * diffTestImage.T

    # print projectTestImage.shape, projectImage.shape
    #按照欧式距离计算最匹配的人脸
    distance = []

    for i in range(imageNumber):
        q = projectImage[:, i]
        # temp = numpy.linalg.norm(projectTestImage - q)
        temp = cosDis(projectTestImage, q)
        distance.append(temp)

    minDistance = numpy.min(distance)
    corresponseIndex = distance.index(minDistance)

    # 返回相对应的下标
    return corresponseIndex

def cosDis(vecA, vecB):
    #使用这个方法得到的结果多一个
    return numpy.abs(vecB-vecA).sum()

class PCAFace:
    def __init__(self, trainData, distanceMeasure):
        """
        初始化函数
        :param trainData: 所有的训练图像
        """
        self.trainData = trainData
        #度量连个样本之间的距离的函数，有多种
        self.distanceMeasure = distanceMeasure
        self._pca()

    def _pca(self):
        # 获取imageDataset中的图像数以及每一张图像的所有的像素点的个数
        imageNumber, perImagePixels = self.trainData.shape
        # 按列计算imageDataseset的平均值,这是一个行向量
        self.meanMatrix = numpy.mean(self.trainData, axis=0)
        # 计算imageDataset与meanMatrix之间的差值
        self.diffMatrix = numpy.mat(self.trainData - self.meanMatrix)
        # 计算特征值和特征向量
        eigValues, eigVectors = numpy.linalg.eig(self.diffMatrix * self.diffMatrix.T)
        #通过linalg.eig函数得到的特征向量基本上已经是按照特征值由大到小的顺序排列的
        self.eigFaces = self.diffMatrix.T * eigVectors
        return None

    def recognize(self, testImage):
        """
        对测试图片进行预测
        :param testImage: 需要测试的图片
        :return:
        """
        # 获取特征空间的图片数目以及每张图片的像素的个数
        perImagePixels, imageNumber = self.eigFaces.shape
        # 将每个样本投影到特征空间
        projectImage = self.eigFaces.T * self.diffMatrix.T
        diffTestImage = numpy.mat(testImage - self.meanMatrix)
        # 将测试的图片投影到特征空间里
        projectTestImage = self.eigFaces.T * diffTestImage.T
        distance = []
        for i in range(imageNumber):
            q = projectImage[:, i]
            temp = self.distance(projectTestImage, q)
            distance.append(temp)
        minDistance = numpy.min(distance)
        # print(distance)
        # corresponseIndex = distance.index(minDistance)
        corresponseIndex = numpy.argsort(distance)
        # 返回相对应的下标
        return corresponseIndex[:3]

    def distance(self, vecA, vecB):
        """
        计算两个向量之间的距离，其中还有多种方法来计算两个向量之间的距离，不过这里实现的是比较简单的那种
        :param vecA:
        :param vecB:
        :return:
        """
        return self.distanceMeasure(vecB, vecA)

if __name__ == '__main__':
    import read_attr as RT
    import pca
    import cPickle

    IMAGE_SIZE = (90, 90)
    trainData, trainLabel, testData, testLabel = RT.getFacesData('att_faces/')

    # pcaFace = PCAFace(trainData)

    # trainData = numpy.array([cv2.resize(item, IMAGE_SIZE).ravel() for item in trainData])
    # testData = numpy.array([cv2.resize(item, IMAGE_SIZE).ravel() for item in testData])

    print("Starting calculate the PCA transform matrix----------")
    # lowDMat, reconMat = pca.pca(trainData, 160)

    # print lowDMat.shape, reconMat.shape
    eigFaces, diffMatrix, meanMatrix = eigenfaceCore(trainData)
    
    # eigFaces1, _, _1 = eigenfaceCore(trainData)
    # print eigFaces.shape
    # data = {
    #     'diffMatrix' : diffMatrix,
    #     'meanMatrix' : meanMatrix,
    #     #这是机器学习实战里面得到的转换矩阵
    #     'reconMat' : reconMat,
    #     #这是根据特征值排序得到的转换矩阵
    #     'eigFaces' : eigFaces,
    #     #这是通过阈值得到的转换矩阵
    #     'eigFacesCore' : eigFaces1
    # }
    print("Saving data into file----------")
    # fileName = 'att_face_PCA_data.txt'
    # fs = open(fileName, 'w')
    # cPickle.dump(data, fs)
    # fs.close()

    print("Saving data is done-------------")

    print("Calculating the PCA transform matrix is done-------------")
    # print eigFaces.shape
    # eigFaces1, diffMatrix, meanMatrix = eigenfaceCore(trainData)

    #显示特征脸的图像，不过特征图像都是虚数，所以使用opencv显示的时候会出现一些警告的信息
    """
    for i in range(10):
        cv2.imshow('ok', reconMat[i, :].astype(numpy.uint8).reshape(IMAGE_SIZE))
        # cv2.imshow('ok1', eigFaces1.T[i, :].astype(numpy.uint8).reshape((100, 100)))
        cv2.waitKey(0) & 0XFF
    cv2.destroyAllWindows()
    """
    """
    # recognize(testData[0], diffMatrix, meanMatrix, eigFaces)
    count = 0
    for index, testImage in enumerate(testData):
        predict = pcaFace.recognize(testImage)
        if trainLabel[predict] == testLabel[index]:
            count += 1
        cv2.imshow('predict image', trainData[predict].reshape(IMAGE_SIZE))
        cv2.imshow('test image', testImage.reshape(IMAGE_SIZE))
        cv2.waitKey(400) & 0XFF
    cv2.destroyAllWindows()
    print count, len(testData)
    #     print predict
    #     pass
    # print testLabel"""