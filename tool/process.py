#_*_coding:utf-8_*_

import cv2
import numpy
from skimage import morphology
 


#判断r1与r2的包含关系
def _inside( r1, r2 ):
	x1, y1, w1, h1 = r1
	x2, y2, w2, h2 = r2

	if ( x1 > x2 ) and ( y1 > y2 ) and ( x1 + w1 < x2 + w2 ) and ( y1 + h1 < x2 + h2 ):
		return True
	else:
		return False

#将图像中的数字包含起来
def _wrap_digit( rect ):
	x, y, w, h = rect
	padding = 8
	hcenter = x + w / 2
	vcenter = y + h / 2
	if ( h > w ):
		w = h
		x = hcenter - w / 2
	if x < 5:
		x = padding
	else:
		h = w
		y = vcenter - h / 2
	if y < 5:
		y = padding

	return ( x-padding, y - padding, w + 2 * padding, h +2 * padding )


"""
在一组多个矩形的数据当中，将较小的矩形过滤掉，一般情况下找到最大的矩形的变长之后，
只要其他的矩形的边长大于它的1/multiple即可

rects 多个矩形的数据数组
multiple 倍数
"""
def filterSmallRect( rects, multiple=4 ):
	result = []
	wMax = 0
	#找出最大的宽高
	for r in rects:
		x1,y1,w1,h1 = r
		if w1 > wMax:
			wMax = w1

	for r in rects:
		x1,y1,w1,h1 = r
		#print wMax, w1
		#注意，这里的4可能得随时更改
		if w1 * multiple > wMax:
			result.append( r )
	return result


"""
img为灰度图像
"""
def findRoi( img ):
	image,cntrs, hire = cv2.findContours( img.copy(), \
		cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

	rectangles = []

	for c in cntrs:
		r = x, y, w, h = cv2.boundingRect( c )
		x, y, w, h = _wrap_digit( r )

		is__inside = False
		for q in rectangles:
			if _inside( r, q ):
				is__inside = True
				break
		if not is__inside:
			rectangles.append( _wrap_digit( r ) )
	return rectangles


"""
找到感兴趣的区域，不使用正方形包裹，而是简单的矩形
"""
def _findObject( img ):
	image,cntrs, hire = cv2.findContours( img.copy(), \
		cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

	rectangles = []

	for c in cntrs:
		r = x, y, w, h = cv2.boundingRect( c )
		x, y, w, h = _wrap_digit( r )

		is__inside = False
		for q in rectangles:
			if _inside( r, q ):
				is__inside = True
				break
		if not is__inside:
			rectangles.append( r )
	return rectangles

"""
将数字之外的东西去掉，传递进来的图片是最多有一个数字，会有其他的小点掺杂，将这些掺杂去掉
input:
	gray 需要寻找对象的图片（灰度值）
	multip 最大的矩形与最小的矩形之间的最大倍数

return:
	新的图像
"""
def getItem( gray, multip=3 ):
	newImage = numpy.zeros( gray.shape )
	rects = _findObject( gray.copy() )
	rects = filterSmallRect( rects, 3 )
	for r in rects:
		x, y, w, h = r
		newImage[y:y+h, x:x+h] = gray[y:y+h, x:x+h]
	return newImage


"""
将某一个区域的正方形缩放为20*20，并拉平这个矩阵
"""
def turn( img ):
	img = cv2.resize( img.copy(), (20, 20) )
	return img

"""
img为灰度图像
kernelSize表示高斯核的大小，整奇数，默认是5
boxSize表示比较窗口的大小，默认是11
"""
def getBinaryImage(img, kernelSize=5, boxSize=5):
	img = img.copy()
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img1 = cv2.GaussianBlur(img, (kernelSize, kernelSize), 1)
	img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, boxSize, 2)
	return img1


"""
将灰度图像转换成彩色图像，方便查看效果
"""
def cvt2BGR(img):
	edges = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
	return edges

def cvt2GRAY(img) :
	edges = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
	return edges


def imshow(img, nameOfWindow='nameOfWindow'):
	cv2.imshow(nameOfWindow, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

"""
细化图像，该图像输入必须是灰度图像，在这个函数里面将其转换成二值图像0-1，
然后返回细化之后的二值图像。
	input :
		img 
	output : 
		img
"""
def Thin( img ):
	img1 = img.copy()
	img1[numpy.where( img1[:, :] > 0 )] = 1
	#
	# 实施骨架算法
	#
	skeleton = morphology.skeletonize( img1 )

	two = numpy.zeros( img1.shape )

	#
	# morphology.skeletonize函数返回的是一个True和False的数组，所以将其转换成0-1的二值图像
	#
	two[numpy.where( skeleton[:, :] == True )] = 1
	two[numpy.where( skeleton[:, :] == False )] = 0

	return two