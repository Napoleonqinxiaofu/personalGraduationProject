#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/20-15:02
description : 非最大抑制
"""
import numpy

def nonMaxSuppressionFast(boxes, overlapThresh):
	#如果没有任何的矩形，那么返回空的列表
	if len(boxes) == 0:
		return []

	#将boxes的数据类型转换成float的类型，如果它们不是,dtype.kind是numpy的属性
	if boxes.dtype.kind == 'i':
		boxes = boxes.astype('float')

	#初始化一些参数，用来挑选最好的矩形
	pick = []

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	#当前矩形的评分
	score = boxes[:, 4]

	#计算这些矩形的面积
	area = (x2-x1+1) * (y2-y1+1)

	idxs = numpy.argsort(score)[::-1]

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		#寻找从当前的点到最后一个矩形之间的最大或者最小的矩形
		xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
		yy1 = numpy.maximum(y1[i], y1[idxs[:last]])

		xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
		yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

		w = numpy.maximum(0, xx2-xx1+1)
		h = numpy.maximum(0, yy2-yy1+1)

		#这他妈计算的是什么呀
		overlap = (w*h) / area[idxs[:last]]

		idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap>overlapThresh)[0])))

		return boxes[pick].astype('int')