#_*_coding:utf-8_*_
########
#author:xiaofu.qin
#description:使用sklearn的svm来训练识别LBP图像
########
from sklearn import svm
#用来保存svm模型的
from sklearn.externals import joblib
import numpy
import cv2
import sys
sys.path.append('../readFaceDataset')
import readDatabase as RD
#引入LBP实现
import sklearn_LBP as SLBP
import LBPClass as CLBP
import distance

trainData, trainLabel, testData, testLabel = RD.getFacesData('../yalefaces/yalefaces', 'sad', True)

testData = None
testLabel = None

lbp = SLBP.LocalBinaryPatterns(16, 2)

allHist = None

#逐一获取每一张图片的lbp图片，然后计算直方图数据
for image in trainData:
	img = lbp.descript(image)
	hist = distance.splitBlock(img)
	if allHist is None:
		allHist = numpy.array(hist).astype('float32')
	else:
		allHist = numpy.vstack((allHist, hist))

#nomalize the data, maybe we do not need it
# allHist = distance.nomalize(allHist)

print( 'Starting train the data')
model = svm.SVC(decision_function_shape='ovo')
model.fit(allHist, trainLabel) 

print("Starting save the model")
joblib.dump(model, 'yaleface_svm_model.pkl')

# X = [[i] for i in range(4)]
# X = numpy.array(X).reshape(len(X), -1)
# Y = [0, 1, 2, 3]
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(X, Y) 