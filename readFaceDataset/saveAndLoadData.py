#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import cPickle

def load( fileName ):
    fs = open( fileName, 'r' )
    data = cPickle.load( fs )
    fs.close()
    lables = numpy.array( [ item[1] for item in data] ).astype( numpy.int32 )
    data = numpy.array([item[0] for item in data])
    return data, lables


def save( data, fileName ):
    fs = open( fileName, 'w' )
    cPickle.dump( data, fs )
    fs.close()
    return None

if __name__ == '__main__':
    import cv2
    import process
    import numpy

    # trainData, trainLabels = load( 'trainData.txt' )
    # testData, testLabels = load( 'testData.txt' )
    #
    # recognizer = cv2.createLBPHFaceRecognizer()
    # recognizer.train( trainData, trainLabels )
    #
    # # Predict of test data
    # for i, testImage in enumerate(testData):
    #     label, confidence = recognizer.predict(testImage)
    #     if label == testLabels[i]:
    #         print "%d 预测成功，置信度为%d" % (label, confidence)
    #     else:
    #         print "预测不成功，原来的标签是%d，预测的标签是%d，置信度为%d" % (testLabels[i], label, confidence)
    #     cv2.imshow("Current image", testImage)
    #     cv2.waitKey(400) & 0xff
    #
    # cv2.destroyAllWindows()

    fs = open('beatyDataSet.txt', 'r')
    data = cPickle.load(fs)
    fs.close()
    for item in data:
        for innerItem in item:
            try:
                cv2.imshow( 'asd', innerItem )
                cv2.waitKey( 0 )
                cv2.destroyAllWindows()
            except Exception as e:
                pass