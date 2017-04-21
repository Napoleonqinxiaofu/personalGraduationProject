#_*_coding:utf-8_*_
########
#author:xiaofu.qin
#description:一些用来计算LBP距离的方法
########
import numpy
import cv2

def calcHist(img, bins=10):
    """
    计算某一张图片的直方数据，如果bins过小的话，有的像素值区域的像素点的个数可能为零，所以为了避免以后计算
    距离的时候出现错误，bins别太少
    :param img: 需要计算直方图的图片
    :param bins: 多少个区域
    :return:
    """
    hist = cv2.calcHist([img], [0], None, [bins], [0, 20])
    return hist.flatten()

def splitBlock(img, lineNumber=3):
	"""
    将图片分块，并计算直方数据，希望图像的宽和高都是偶数，这样分隔起来比较好
    :param img: 需要分块的图片
    :param lineNumber : 将图像分成(lineNumber+1)**2个小的区域来计算直方图数据
    :return:
	"""
	h, w = img.shape[:2]

	W = numpy.arange(0, w, w * 1.0 / (lineNumber + 1)).astype('uint8')
	H = numpy.arange(0, h, h * 1.0 / (lineNumber + 1)).astype('uint8')

	splitW = numpy.hstack((W, w)).astype('uint8')
	splitH = numpy.hstack((H, h)).astype('uint8')

	histInfo = []

	for indexX, x in enumerate(W):
		for indexY, y in enumerate(H):
			try:
				hist = calcHist(img[y:splitH[indexY + 1], x:splitH[indexX + 1]], 10)
			except Exception as err:
				print(err)
				hist = []
			histInfo.extend(hist)

	return histInfo

def nomalize(hists):
	hists = numpy.array(hists)
	minValues = numpy.min(hists, axis=0)
	maxValues = numpy.max(hists, axis=0)
	rangeValues = maxValues - minValues
	return numpy.float32(hists-minValues) / rangeValues

def setNanToZero(arr):
	arr = numpy.array(arr)
	arr[numpy.where(numpy.isnan(arr))] = 0
	return arr

#|x-y|,这表示x与y向量之差的模
def L1(hist1, hist2):
	hist1 = setNanToZero(hist1)
	hist2 = setNanToZero(hist2)
	return numpy.sqrt(numpy.sum((hist1-hist2)**2))

#||x-y||^2
def Euler(hist1, hist2):
	hist1 = setNanToZero(hist1)
	hist2 = setNanToZero(hist2)
	return numpy.linalg.norm(hist1-hist2)

#余弦距离
def cosine(hist1, hist2):
	hist1 = setNanToZero(hist1)
	hist2 = setNanToZero(hist2)

	return -(hist1 * hist2.T).sum() / (numpy.linalg.norm(hist1) * numpy.linalg.norm(hist2))
