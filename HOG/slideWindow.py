#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/20-11:24
description : 滑动窗口
"""
import numpy

def slideWindow(img, stepSize, windowSize):
	for y in numpy.arange(0, img.shape[0], stepSize):
		for x in numpy.arange(0, img.shape[1], stepSize):
			yield (x, y, img[y:y+windowSize[1], x:x+windowSize[0]])