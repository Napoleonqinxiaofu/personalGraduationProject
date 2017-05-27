#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever

全局注释：LBPH算法与FisherFace、EigenFace算法的置信度评级不是一个标准，后两者的评分将产生0~20000的值，
        所以他们产生了4000~5000之间的置信度评分都算是比较可靠的。
        LBPH算法产生的评分就比较低，低于50的被称为好的分数，高于80的被称为不好的评分
"""
import cv2
import numpy
from PIL import Image
import os
import pickle

def getFacesData( folderPath, testExtention, isDebug=False, IMAGE_SIZE = (90, 90) ):
    """
    获取yalefaces数据库的图像
    :param folderPath: 数据库的目录
    :param testExtendion: 用来作为测试的图片的后缀
    :return:
    """
    cascadePath = "../haar/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    imagesPath = [os.path.join( folderPath, filename ) for filename in os.listdir( folderPath )\
                  if not filename.endswith( 'txt' ) and not filename.endswith( 'gif' ) ]
    # print imagesPath

    images = []
    testImages = []
    labels = []
    testLabel = []
    
    for filename in imagesPath:
        if isDebug:
            print(filename)
        img = Image.open( filename ).convert( 'L' )
        img = numpy.array( img, dtype=numpy.uint8 )

        #start detect the face in image
        rects = faceCascade.detectMultiScale(img, 1.1, 3)
        for r in rects:
            x, y, w, h = r
            image = cv2.resize(img[y:y+h, x:x+w], IMAGE_SIZE)
            label = int(os.path.split(filename)[1].split(".")[0].replace("subject", ""))
            if testExtention in filename:
                testImages.append( image )
                testLabel.append( label )
            else:
                images.append( image )
                labels.append( label )

    return images, labels, testImages, testLabel

if __name__ == '__main__':
    import read_feret as RT
    IMAGE_SIZE = (90, 90)
    IMAGE_SIZE1 = (60, 60)
    trainData, trainLabels, testData, testLabels = RT.getFacesData('FERET/')
    trainData = numpy.array([cv2.resize(data.reshape(IMAGE_SIZE), IMAGE_SIZE1) for data in trainData])
    testData = numpy.array([cv2.resize(data.reshape(IMAGE_SIZE), IMAGE_SIZE1) for data in testData])

    #Starting ti training the data
    #Initialize of face recognizer
    #LBHF算法不将图像归一化，有一个识别错误。但是将图像尺寸归一化之后的能全部识别
    # recognizer = cv2.createLBPHFaceRecognizer()

    #Eigenface 就需要将图像尺寸归一化，全部识别正确
    recognizer = cv2.createEigenFaceRecognizer()

    #有一个预测失败
    # recognizer = cv2.createFisherFaceRecognizer()

    recognizer.train(numpy.array(trainData), numpy.array( trainLabels, dtype=numpy.int32 ))

    #Predict of test data
    count = 0
    for i, testImage in enumerate(testData):
        label, confidence = recognizer.predict(testImage)
        if label == testLabels[i]:
            print("%d 预测成功，置信度为%d" % ( label, confidence))
            count += 1
        else:
            print("预测不成功，原来的标签是%d，预测的标签是%d，置信度为%d" % (testLabels[i], label, confidence))
    #     cv2.imshow("Current image", testImage)
    #     cv2.waitKey(400) & 0xff
    #
    # cv2.destroyAllWindows()
    print(count)
    #保存数据，方便下次直接提取，已经保存过了
    # fs = open( 'trainData.txt', 'w' )
    # cPickle.dump( zip( trainData, trainLabels ), fs )
    # fs.close()
    #
    # fs = open( 'testData.txt', 'w' )
    # cPickle.dump( zip( testData, testLabels ), fs )
    # fs.close()