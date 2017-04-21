#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
"""
实现LBP算法,使用skimage，因为opencv提供的LBP算法的层次比较高，我们可以使用它来进行人脸识别，但是对于普通的纹理特征提取它却没有
提供相关的操作，所以我们只能使用skimage库
"""
import numpy
import cv2
from PIL import Image
import os
from skimage import feature
from sklearn.svm import LinearSVC

import process

class LBP:
    def __init__(self, numPoints=8, radius=1):
        self.numPoints = numPoints
        self.radius = radius

    def descripe(self, image, eps = 1e-6):
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints,
                                            R=self.radius, method='uniform')

        #计算直方图,lbp的值的范围是0~self.numPoints+2之间，这是为什么？
        (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(0, self.numPoints+3),
                                     range=(0, self.numPoints+2))
        hist = hist.astype(numpy.float32)
        #normalize hist
        hist /= (numpy.sum(hist)+eps)
        return hist

    #识别函数，第一个参数为需要进行识别的LBP的特征，第二个参数为已有的所有的训练的数据的LBP特征以及标签的SVM实例
    def predict(self, lbpdesc, svmModel):
        return svmModel.predict(lbpdesc)[0]

def train(imageFolder, LBPInstance):
    if LBPInstance is None:
        return None
    data = []
    lables = []
    imageFolder = imageFolder if imageFolder.endswith( '/' ) else imageFolder + '/'
    for filename in os.listdir( imageFolder ):
        if filename.endswith( 'sad' ):
            continue
        img = Image.open( imageFolder + filename )
        img = img.convert( 'L' )
        lbpDesc = LBPInstance.descripe( img )
        data.append( lbpDesc )
        lables.append( filename.split('.')[0] )
    # train a Linear SVM on the data
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, lables)
    return model


def train1(imageFolder, LBPInstance ):
    if LBPInstance is None:
        return None
    data = []
    lables = []
    imageFolder = imageFolder if imageFolder.endswith( '/' ) else imageFolder + '/'
    for filename in os.listdir( imageFolder ):
        img = cv2.imread( imageFolder + filename )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        data.append( LBPInstance.descripte( img ) )
        lables.append( filename.split('.')[0] )

    # train a Linear SVM on the data
    model = LinearSVC(C=100.0, random_state=42)
    model.fit( data, lables )
    return model


if __name__ == '__main__':
    # imagePath = './image/4.jpg'
    # img = cv2.imread( imagePath )
    # gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    lbp = LBP( 24, 5 )
    model = train( './yalefaces/yalefaces/', lbp )

    #这是在没有提取脸部的情况下进行的测试，错误率为2/15
    testData = []
    predict = []
    for filename in os.listdir( './yalefaces/yalefaces/' ):
        if not filename.endswith( 'sad' ):
            continue
        img = Image.open('./yalefaces/yalefaces/' + filename)
        img = img.convert('L')
        testData.append( filename )
        lbpDesc = lbp.descripe(img)
        # lbpDesc = lbpDesc.reshape( len(lbpDesc), -1 )
        predict.append( lbp.predict(lbpDesc, model) )

    print zip( testData, predict )
