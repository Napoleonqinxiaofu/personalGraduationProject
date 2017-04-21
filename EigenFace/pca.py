#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import numpy

"""
实现pca数据降维
inputs:
    dataMatrix : 原始数据矩阵
    featureNumber : 降维之后的举个特征值，我们需要多少个就传递进去多少个
"""
def pca( dataMatrix, featureNumber=1000 ):
    #去平均值，下面的语句是对dataMatrix矩阵的每一列取平均值
    meanValues = numpy.mean( dataMatrix, axis=0 )
    #dataMatrix矩阵的每一列都减去meanValues矩阵的相对应的值，比如第一列就会减去meanValues矩阵的第一个元素
    removeMeanValues = dataMatrix - meanValues

    #计算协方差矩阵，该矩阵的维度与dataMatrix一致
    covMatrix = numpy.cov( removeMeanValues, rowvar=0 )

    #计算特征值与特征向量
    eigValues, eigVector = numpy.linalg.eig( numpy.mat( covMatrix ) )

    eigValuesInd = numpy.argsort( eigValues )
    eigValuesInd = eigValuesInd[:-(featureNumber+1):-1]

    redEigVector = eigVector[:, eigValuesInd]

    lowDDataMat = removeMeanValues * redEigVector
    reconMat = ( lowDDataMat * redEigVector.T ) + meanValues
    #reconMat是PCA变换矩阵
    return lowDDataMat, reconMat

def pca_bate(data, k):
    data = numpy.float32(numpy.mat(data))
    height, width = data.shape
    #对列求均值
    meanVector = numpy.mean(data, axis=0)
    #让meanVector在高度方向想复制height-1次，在宽度方向上不进行复制
    meanMatrix = numpy.tile(meanVector, (height, 1))

    removeMean = data - meanMatrix

    T1 = removeMean * removeMean.T

    eigValue, eigVector = numpy.linalg.eig(T1)

    V1 = eigVector[:, 0:k]  # 取前k个特征向量
    V1 = removeMean.T * V1
    for i in xrange(k):  # 特征向量归一化
        L = numpy.linalg.norm(V1[:, i])
    V1[:, i] = V1[:, i] / L
    data_new = removeMean * V1  # 降维后的数据
    return data_new, meanVector, V1


def loadData( filename ):
    fr = open( filename )
    stringArr = [line.strip().split( "\t" ) for line in fr.readlines()]

    #map函数仅仅是将line这个变量里面的数据编程float类型的而已
    #这个函数不是什么重要的转换类型的函数，只不过它比直接的float函数好的地方在于可以直接传递一个数组进去
    dataArr = [map(float, line) for line in stringArr ]

    return numpy.mat( dataArr )


if __name__ == '__main__':
    pass