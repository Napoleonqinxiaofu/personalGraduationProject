#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
在面对sklearn的LBP的测试效果还是差不多的，只不过使用框架的速度比较快
"""
from skimage import feature
import numpy
import cv2

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def descript(self, image, eps=1e-7):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        return lbp.astype('float32')

if __name__ == '__main__':
    import  cv2
    import pickle
    img = cv2.imread('../image/4.jpg', 0)

    lbp = LocalBinaryPatterns(16, 2)

    lbpImage = lbp.describe(img)
    print(numpy.max(lbpImage))

    print(cv2.calcHist([lbpImage], [0], None, [5], [0, 10]))
    cv2.imshow('origin image', img)
    cv2.imshow('sklearn lbp image', lbpImage)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()