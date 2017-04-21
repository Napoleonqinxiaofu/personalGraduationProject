#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
decription:LBP方法的类封装
"""
import cv2
import numpy

class LBP:
    def __init__(self, P=16, R=2):
        """
        初始化函数
        :param P: 需要多少个点
        :param R:  这些点的半径取多少
        """
        self.P = P
        self.R = R

        # 定义比较函数
        self._s = lambda x, y: 1 if x > y else 0

        # 定义获取LBP算子的坐标，这些坐标有极大的可能就是小数，所以要使用双线性插值来获取确定已经存在的像素值
        self._coordX = lambda xc, p: xc + self.R * numpy.cos(2 * numpy.pi * p / self.P)
        self._coordY = lambda yc, p: yc - self.R * numpy.sin(2 * numpy.pi * p / self.P)

        # 获得所有的坐标
        self._getAllCoords = lambda x, y: [(self._coordX(x, i), self._coordY(y, i)) for i in range(self.P)]

    def descript(self, img):
        """
        获取某张图片的LBP响应图片
        :param img: 需要计算的原图像
        :return:
        """
        # 定义一个LBP纹理图像
        LBPImage = numpy.zeros(img.shape).astype(numpy.uint8)

        h, w = img.shape

        for x in range(w):
            for y in range(h):
                LBPImage[y, x] = self._getSingleLBP(img, x, y)

        print("Done")
        return LBPImage


    def _getSingleLBP(self, img, x, y):
        """
        获取(x,y)点的响应LBP算子纹理值
        :param img:
        :param x:
        :param y:
        :return:
        """
        allCoords = self._getAllCoords(x, y)

        # 获取所有的相邻的像素点的像素值
        allGrayValues = [self._getGrayScaleValue(img, ix, iy) for ix, iy in allCoords]

        # 将获得的像素值与中心的像素值进行比较
        allBinary = [self._s(img[y, x], grayValue) for grayValue in allGrayValues]

        # 下面的数组第一个数代表p为0的时候的比较值
        return self._findMinValue(allBinary)

    def _findMinValue(self, binary):
        """
               寻找获取一串儿二进制的循环排列最小的值
               :param binaryArray:
               :return:
               """
        length = len(binary)
        values = []
        for i in range(length):
            binary.append(binary.pop(0))
            newBinary = [str(item) for item in binary]
            values.append(int(''.join(newBinary), 2))

        return min(values)

    def _getGrayScaleValue(self, img, x, y):
        """
        获取图像之中的某一点的像素值，如果这个点在图像上不存在，那么以0代替。
        :param img :
        :param x: x坐标
        :param y: y坐标
        :return:
        """
        returnValue = None
        try:
            returnValue = img[int(y), int(x)]
        except:
            returnValue = 0
        return returnValue

    def _doubleLinear(self, img, x, y):
        """
        双线性插值法
        :param x: 目标像素点的x坐标
        :param y: 目标像素点的y坐标
        :return: 目标像素点进行插值计算之后的像素值
        """
        intX = int(numpy.floor(x))
        intY = int(numpy.floor(y))

        greatIntX = intX + 1
        greatIntY = intY + 1

        gapX = x - intX
        gapY = y - intY

        g00 = self._getGrayScaleValue(img, intY, intY)
        g01 = self._getGrayScaleValue(img, greatIntX, intY)
        g10 = self._getGrayScaleValue(img, intX, greatIntY)
        g11 = self._getGrayScaleValue(img, greatIntX, greatIntY)

        gxy = g00*(1-gapX)*(1-gapY) + g10*gapX*(1-gapY) + g01*(1-gapX)*gapY + g11*gapX*gapY

        #也可以使用矩阵的形式
        #好像不可以使用矩阵的形式，因为严格的矩阵运算得到的结果比较大
        # gxy = numpy.mat([1-gapX, x]) * numpy.mat([[g00, g01], [g10, g11]]) * numpy.mat([1-gapY, gapY]).T
        if gxy < 0:
            gxy = 0
        elif gxy > 255:
            gxy = 255

        return gxy

#LBPFace的改进版，这个类需要传递的是已经计算好了的直方数据，而不是图片
class HistCompare:
    def __init__(self, trainHists):
        self.trainHists = trainHists

    def recognize(self, hist, distanceFunc):
        values = []
        for train in self.trainHists:
            #有好多值是nan的情况，所以要警惕
            values.append(distanceFunc(train, hist))
        return numpy.argsort(values)


class LBPFace:
    def __init__(self, trainLBPData, areaNumber=9):
        """
        这个方法需要提前计算所有训练图像以及测试图像的LBP纹理图像
        :param allTrainHist: 所有的训练图像的直方图数据
        :param areaNumber : 即将把整张图片分成areaNumber个小的区域
        """
        # self.trainLBPData = trainLBPData
        self.lineNumber = int(numpy.sqrt(areaNumber))-1
        self.AllTrainingHist = [self._splitBlock(item, self.lineNumber)
                                for item in trainLBPData]

    def recognize(self, testImage):
        """
        进行人脸预测的函数（其实不光是人脸预测，也可以是其他的图片
        :param testHist: 需要进行预测的图像
        :return:
        """
        # 每张图片有h个直方数据
        testHist = self._splitBlock(testImage, self.lineNumber)
        AllChi = []
        for hist in self.AllTrainingHist:
            AllChi.append(self._Chi(hist, testHist))

        return AllChi.index(min(AllChi))

    def _Chi(self, hist1, hist2):
        """
            计算直方图数据的Chi距离
            :param hist1:
            :param hist2:
            :return:
            """
        value = 0
        for index, item in enumerate(hist1):
            #防止分母为零的情况
            try:
                a = (hist2[index]-item)**2 / (hist2[index]+item)
            except:
                a = 0
            value += a
        return value


    def _splitBlock(self, img, lineNumber=3):
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

        histInfo = None

        for indexX, x in enumerate(W):
            for indexY, y in enumerate(H):
                hist = self._calcHist(img[y:splitH[indexY + 1], x:splitH[indexX + 1]], 20)
                if histInfo is None:
                    histInfo = hist
                else:
                    histInfo = numpy.hstack((histInfo, hist))

        return histInfo


    def _calcHist(self, img, bins=10):
        """
        计算某一张图片的直方数据，如果bins过小的话，有的像素值区域的像素点的个数可能为零，所以为了避免以后计算
        距离的时候出现错误，bins别太少
        :param img: 需要计算直方图的图片
        :param bins: 多少个区域
        :return:
        """
        hist = cv2.calcHist([img], [0], None, [bins], [0, 20])
        return hist.flatten()


if __name__ == '__main__':
    import pickle

    with open('../dlib/myBeautyLBPData.txt', 'rb') as fs:
        data = pickle.load(fs)
        fs.close()

    face = LBPFace(data['trainData'], 16)
    count = 0
    for index, image in enumerate(data['testData']):
        predict = face.recognize(image)
        print("The predict label is : %d, and the test label is : %s" % (data['trainLabel'][predict], data['testLabel'][index]))

        if( data['trainLabel'][predict] == data['testLabel'][index]):
            count += 1
        cv2.imshow('predict image', data['originTrainData'][predict])
        cv2.imshow('test image', data['originTestData'][index])
        cv2.waitKey(0) & 0XFF
    cv2.destroyAllWindows()
    print( count )】
