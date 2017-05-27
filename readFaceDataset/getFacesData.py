#_*_coding:utf-8_*_
########
# author : xiaofu.qin
# description: 封装读取yalefaces、att、feret数据库图片的函数
########
import readDatabase as RD
import read_attr as RA
import read_feret as RF
import readMyBeauty as RB
import cv2

"""
@param dataType {string} {yalefaces att feret myBeauty}
@param folder {string} {图片存放的文件夹}
@param isDebug {bool} {是否显示当前处理到什么进度}
@param isGray {bool} {是否需要灰度图片，默认为需要}
@param imageNumber {number} {需要获取的图片数量，只对feret数据库管用}
"""
def getFacesData(dataType, folder, isDebug=False, isGray=True, imageNumber=50, IMAGE_SIZE=(90, 90)):
	trainData, trainLabel, testData, testLabel = None, None, None, None
	if dataType == 'yalefaces':
		#默认yalefaces的测试图片后缀为sad的图片
		trainData, trainLabel, testData, testLabel = RD.getFacesData(folder, 'sad', isDebug, IMAGE_SIZE=IMAGE_SIZE)
	elif dataType == 'att':
		trainData, trainLabel, testData, testLabel = RA.getFacesData(folder, isDebug, IMAGE_SIZE)
	elif dataType == 'feret':
		trainData, trainLabel, testData, testLabel = RF.getFacesData(folder, imageNumber, isDebug, IMAGE_SIZE)
	elif dataType == 'myBeauty':
		#读取自己的数据库不需要将图片尺寸归一化，因为要拿给dlib来识别，不太用的上
		width = 200 if IMAGE_SIZE[0] == 90 else IMAGE_SIZE[0]
		trainData, trainLabel, testData, testLabel = RB.getFacesData(folder, isDebug, width=width)

	if not isGray:
		trainData = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in trainData]
		testData = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in testData]

	return trainData, trainLabel, testData, testLabel