#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
"""
import cv2
import numpy
import sys

sys.path.append('../readFaceDataset')
import readDatabase as RD
import read_attr as RF

import eigenfaces

sys.path.append('../LBP')
import distance


# trainData, trainLabel, testData, testLabel = RD.getFacesData('../yalefaces/yalefaces', 'sad', True)
trainData, trainLabel, testData, testLabel = RF.getFacesData('../att_faces', True)

IMAGE_SIZE = (90, 90)

trainData = numpy.array([(cv2.resize(data, IMAGE_SIZE)).flatten() for data in trainData])
testData = numpy.array([(cv2.resize(data, IMAGE_SIZE)).flatten() for data in testData])

#实例化EigenFaces
PCAFace = eigenfaces.PCAFace(trainData, distance.Euler)
correctCount = 0
for index, image in enumerate(testData):
	predict = PCAFace.recognize(image)
	if testLabel[index] == trainLabel[predict[0]]:
		correctCount += 1
	# cv2.imshow('img', numpy.hstack([image.reshape(IMAGE_SIZE), trainData[predict[0]].reshape(IMAGE_SIZE), trainData[predict[1]].reshape(IMAGE_SIZE), trainData[predict[2]].reshape(IMAGE_SIZE)]))
	# cv2.waitKey(400) & 0XFF
cv2.destroyAllWindows()

print("The accuracy is %d/%d-----%f" % (correctCount, len(testData), float(correctCount)/len(testData)))


# for i in range(3):
#     cv2.imshow('img', trainData[i].reshape(IMAGE_SIZE))
#     cv2.waitKey(1000)

# cv2.destroyAllWindows()
"""
label = 0

wholeWraper = []
for i, imageLabel in enumerate(trainLabels):
    if label == imageLabel:
        singleWraper = numpy.vstack((singleWraper,trainData[i]))
    else:
        try:
            wholeWraper.append(singleWraper)
        except Exception as e:
            print e
        singleWraper = trainData[i]
        label = imageLabel

low, rec = pca.pca( wholeWraper[0][0], 100)
print low.shape
"""
# allImage = []
# for i in range(3):
#     img = trainData[i].reshape((150, 150))
#     corners = cv2.goodFeaturesToTrack(img, 20, 0.04, 10)
#     for corner in corners:
#         x, y = corner.ravel()
#         cv2.circle(img, (x,y), 3, 255, -1)
#     if len(allImage) == 0:
#         allImage = numpy.hstack([img])
#     else:
#         allImage = numpy.hstack([
#             allImage, img
#         ])
# cv2.imshow('img', allImage)
# cv2.waitKey(0) & 0XFF
#
# cv2.destroyAllWindows()
