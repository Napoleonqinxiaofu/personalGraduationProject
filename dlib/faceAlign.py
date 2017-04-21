#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/23-10:30
description : 
"""
import math
import cv2

#
# 旋转图像
#
def rotate( img, centerPoint, angle, scale=1.0 ):
    matrix = cv2.getRotationMatrix2D(centerPoint, angle, scale)
    w = img.shape[1]
    h = img.shape[0]
    rotationImage = cv2.warpAffine(img, matrix, (w, h))

    return rotationImage

class FaceAlign:
    def __init__(self):
        self.angle = None
	#
    def align(self, img, right, left):
        self.calcAngle(right, left)

        centerPoint = (img.shape[0] // 2, img.shape[1] // 2)
        img = rotate(img, centerPoint, self.angle)
        return img

    def getAngle(self):
        return self.angle

    def calcAngle(self, right, left):
        angle = float(right[1] - left[1]) / (right[0] - left[0])
        angle = math.degrees(math.atan(angle))
        self.angle = angle


if __name__ == '__main__':
    imgPath = './image/1.jpg'