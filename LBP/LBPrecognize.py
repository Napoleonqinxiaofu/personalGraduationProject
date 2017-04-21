#_*_coding:utf-8_*_
"""
明天的任务：整顿一下AllChi数组里面都是nan的问题（顺便说一下，numpy数组的逐次操作也不是什么好的东西
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
description : LBP算法的实现
"""
import cv2
import numpy

def LBP(img, P=8, R=1):
    """
    LBP算法的主过程，边界暂时还不知道如何处理，所以不会处理，到时候有网了再去查
    :param img: 需要计算LBP纹理特征的图像
    :param P: 多少个点
    :param R: 半径
    :return: 经过计算了的LBP纹理图片
    """
    #定义比较函数
    _s = lambda x, y : 1 if x>y else 0

    #定义获取LBP算子的坐标，这些坐标有极大的可能就是小数，所以要使用双线性插值来获取确定已经存在的像素值
    _coordX = lambda xc, p : xc + R*numpy.cos(2*numpy.pi*p/P)
    _coordY = lambda yc, p : yc - R*numpy.sin(2*numpy.pi*p/P)

    #获得所有的坐标
    _getAllCoords = lambda x, y : [(_coordX(x, i), _coordY(y, i)) for i in range(P)]


    def _getGrayScaleValue(x, y):
        """
        获取图像之中的某一点的像素值，如果这个点在图像上不存在，那么以0代替。
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



    def _doubleLinear(x, y):
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

        g00 = _getGrayScaleValue(intY, intY)
        g01 = _getGrayScaleValue(greatIntX, intY)
        g10 = _getGrayScaleValue(intX, greatIntY)
        g11 = _getGrayScaleValue(greatIntX, greatIntY)

        gxy = g00*(1-gapX)*(1-gapY) + g10*gapX*(1-gapY) + g01*(1-gapX)*gapY + g11*gapX*gapY

        #也可以使用矩阵的形式
        #好像不可以使用矩阵的形式，因为严格的矩阵运算得到的结果比较大
        # gxy = numpy.mat([1-gapX, x]) * numpy.mat([[g00, g01], [g10, g11]]) * numpy.mat([1-gapY, gapY]).T
        if gxy < 0:
            gxy = 0
        elif gxy > 255:
            gxy = 255
        return gxy


    def _findMinValue(binaryArray):
        """
        寻找获取一串儿二进制的循环排列最小的值
        :param binaryArray:
        :return:
        """
        length = len(binaryArray)
        values = []
        for i in range(length):
            binaryArray.append(binaryArray.pop(0))
            newBinaryArray = [str(item) for item in binaryArray]
            values.append(int(''.join(newBinaryArray), 2))

        return min(values)


    #获取某一个像素点最小的LBP算子
    def _getSingleLBP(x, y):
        allCoords = _getAllCoords(x, y)

        #获取所有的相邻的像素点的像素值
        allGrayValues = [_getGrayScaleValue(ix, iy) for ix, iy in allCoords]

        #将获得的像素值与中心的像素值进行比较
        allBinary = [_s(img[y, x], grayValue) for grayValue in allGrayValues]

        #下面的数组第一个数代表p为0的时候的比较值
        return _findMinValue(allBinary)

    #定义一个LBP纹理图像
    LBPImage = numpy.zeros(img.shape).astype(numpy.uint8)

    h, w = img.shape

    for x in range(w):
        for y in range(h):
            LBPImage[y, x] = _getSingleLBP(x, y)

    print "Done"
    return LBPImage

def splitBlock(img, hLineNumber=3, vLineNumber=3):
    """
    将图片分块，并计算直方数据，希望图像的宽和高都是偶数，这样分隔起来比较好
    :param img: 需要分块的图片
    :param hLineNumber: 水平直线的数量
    :param vLineNumber: 垂直直线的数量
    :return:
    """
    h, w = img.shape[:2]

    W = numpy.arange(0, w, w*1.0/(hLineNumber+1)).astype('uint8')
    H = numpy.arange(0, h, h*1.0/(vLineNumber+1)).astype('uint8')

    splitW = numpy.hstack((W, w)).astype('uint8')
    splitH = numpy.hstack((H, h)).astype('uint8')

    histInfo = None

    for indexX, x in enumerate(W):
        for indexY, y in enumerate(H):
            hist = calcHist(img[y:splitH[indexY+1], x:splitH[indexX+1]], 30)
            if histInfo is None:
                histInfo = hist
            else:
                histInfo = numpy.hstack((histInfo, hist))

    return histInfo


def calcHist(img, bins=10):
    """
    计算某一张图片的直方数据，如果bins过小的话，有的像素值区域的像素点的个数可能为零，所以为了避免以后计算
    距离的时候出现错误，bins别太少
    :param img: 需要计算直方图的图片
    :param bins: 多少个区域
    :return:
    """
    hist = cv2.calcHist([img], [0], None, [bins], [0, 50])
    return hist.ravel()

def Chi(hist1, hist2):
    """
    计算直方图数据的Chi距离，注意防止分母为零的情况
    :param hist1:
    :param hist2:
    :return:
    """
    value = 0
    for index, item in enumerate(hist1):
        a = (hist2[index]-item)**2/(hist2[index]+item)
        if a == a:
            value += a
        # try:
        #     value += (hist2[index]-item)**2/float(hist2[index]+item)
        # except Exception as e:
        #     print e
    return value


def recognize(AllTrainingHist, testHist, k=6):
    """
    计算测试图片与人脸数据库之中的图片的直方数据的距离，使用Chi平方的方法
    :param AllHist: 所有的人脸数据库图片的直方数据 m×h1
    :param testHist: 测试使用的图片的直方图数据 1*h2
    :return:
    """
    #每张图片有h个直方数据
    AllChi = []
    for hist in AllTrainingHist:
        AllChi.append(Chi(hist, testHist))

    minValue = min(AllChi)
    minIndex = numpy.argsort(AllChi)

    #最邻近的几个距离
    nearChi = numpy.array(AllChi)[minIndex[:k]]

    indexs = []
    for near in nearChi:
        indexs.append(AllChi.index(near))

    return AllChi.index(minValue), indexs


def filter(trainLabel, indeces):
    """
    从indeces给出的下表中判断最优的预测标签
    首先如果各个标签各不相同，那么取第一个
    如果有多个标签，那么取数量最多的标签
    如果最多的几个标签之间的数量相同，那么比较第一个，哪一个的距离更小，就是用哪一个
    :param indeces: 下标的列表
    :return:
    """
    lables = [trainLabel[index] for index in indeces]
    count = countLabel(lables)
    values = count.values()
    # print count
    maxV = max(values)
    length = len(values)

    if length == len(lables):
        return lables[0]
    #如果count里面记录的values只有一个最大值，那么返回该最大值所对应的键值
    elif len(numpy.nonzero(numpy.array(values)-maxV)[0]) == length-1:
        for k, v in count.items():
            if v == maxV:
                return int(k.replace(']', '').replace('[', ''))
    #如果count里面记录的values同时又两个以上的最大值，那么比较这些值的第一个距离，谁小选择谁
    elif len(numpy.nonzero(numpy.array(values)-maxV)[0]) < length-1:
        #记录所对应的keys
        ks = []
        for k, v in count.items():
            if v == maxV:
                ks.append(k)

        #寻找最小距离的key值
        for label in lables:
            if str(label) in ks:
                return int(str(label).replace(']', '').replace('[', ''))
    else:
        return lables[0]


def countLabel(labels):
    count = {}
    for label in labels:
        count[str(label)] = count.get(str(label), 0) + 1
    return count

if __name__ == "__main__":
    img = cv2.imread('./image/1.jpg', 0)
    img2 = cv2.imread('./image/0.jpg', 0)
    # print calcHist(img[:30, :40], 40).ravel().shape

    img = cv2.resize(img, (90, 90))
    img2 = cv2.resize(img2, (90, 90))

    hist1 = splitBlock(img, 3, 3)
    hist2 = splitBlock(img2, 3, 3)
    # print hist1
    print Chi(hist1, hist2)
    # print splitBlock(img, 3, 3).shape

    """
    import readDatabase as RD
    print "Starting reading the image database from disk----------"
    trainData, trainLabel, testData, testLabel = \
        RD.getFacesData('yalefaces/yalefaces/', 'sad')

    #resize the image
    IMAGE_SIZE = (90, 90)

    print "Resizing the image to ", IMAGE_SIZE, '---------------'
    trainData = [cv2.resize(item, IMAGE_SIZE) for item in trainData]
    testData = [cv2.resize(item, IMAGE_SIZE) for item in testData]

    print "Reading all image is done----------------"

    print "Starting extract the LBP feature--------"

    trainLBPData = [LBP(item, 8, 2) for item in trainData]
    testLBPData = [LBP(item, 8, 2) for item in testData]

    print "Extracting the LBP feature is done--------"

    print "Splitting the image into 9 block area and calculating it's histogram--------"
    #现在trainLBPData里面每一个元素都包含9个直方
    trainLBPData = [splitBlock(item, 2, 2) for item in trainLBPData]
    testLBPData = [splitBlock(item, 2, 2) for item in testLBPData]

    print "Calculating the histogram is done---------------"



    for index, image in enumerate(testLBPData):
    # for i in xrange(0, 140, 10):
        predict = recognize(trainLBPData, image)
        print predict
        cv2.imshow("predict image", trainData[predict])
        cv2.imshow("test image", testData[index])
        cv2.waitKey(0) & 0XFF

    cv2.destroyAllWindows()
    """