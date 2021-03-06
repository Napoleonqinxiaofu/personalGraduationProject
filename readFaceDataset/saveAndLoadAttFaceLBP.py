#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import cPickle
import numpy

def load( fileName ):
    fs = open(fileName, 'r')
    data = cPickle.load(fs)
    fs.close()
    return data


def save(data, fileName):
    fs = open(fileName, 'w')
    cPickle.dump(data, fs)
    fs.close()
    return None

if __name__ == "__main__":
    #保存我们的LBP算子图片
    import read_attr as RA
    import cv2
    import LBPrecognize as LBP
    import sklearn_LBP as SL

    lbp = SL.LocalBinaryPatterns(16, 2)

    print "Starting reading the image database from disk----------"
    trainData, trainLabel, testData, testLabel = \
        RA.getFacesData('./att_faces/')

    print "Loading the image from disk is done------------"

    print "Resizing the image--------------"
    IMAGE_SIZE = (90, 90)

    # trainData = [cv2.resize(item, IMAGE_SIZE) for item in trainData]
    # testData = [cv2.resize(item, IMAGE_SIZE) for item in testData]

    print "Starting to extract the LBP feature---------------"

    trainData = [lbp.describe(item.reshape(IMAGE_SIZE)) for item in trainData]
    testData = [lbp.describe(item.reshape(IMAGE_SIZE)) for item in testData]

    print "Extracting the LBP feature is done--------------"

    for i in range(0, 100, 20):
        cv2.imshow('asd', trainData[i])
        cv2.waitKey(0) & 0XFF
    cv2.destroyAllWindows()


    print "Saving the data and labels-----------"
    objDict = {
        "trainData" : trainData,
        "trainLabel" : trainLabel,
        "testData" : testData,
        "testLabel" : testLabel
    }
    save(objDict, 'attFaceLBP_sklearn.txt')

    print "Saving done!--------------"