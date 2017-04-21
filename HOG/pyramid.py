#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/20-11:19
description : 图像金字塔
"""
import cv2

def resize( img, scale):
	return cv2.resize(img, (int(img.shape[0] * (1 / scale)), int(img.shape[1] * (1/scale))), interpolation=cv2.INTER_AREA)

def pyramid(img, scale=1.5, minScale=(200, 80)):
	yield img

	while True:
		img = resize(img, scale)
		if( img.shape[0] < minScale[1] or img.shape[1] < minScale[0]):
			break;
		yield img
