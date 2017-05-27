#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:whatever
description : 读取从百度图片上获取的图片集合，过滤掉一些不合格的文件
"""
import os
import cv2
import numpy
import dlib
import stat

import random
import string

stringLetters = string.ascii_letters

def randomChars(num=10):
    arr = [random.choice(stringLetters) for i in range(num)]
    return  ''.join(arr)

#判断字符串中是否包含数字
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

#获取dlib检测脸部的功能
detector = dlib.get_frontal_face_detector()

def renameFile(oldFileName, newFileName):
    os.rename(oldFileName, newFileName)

#判断图像中的人脸数量，不等于1则返回True，否则返回False
def judgeFaceNum(imgPath, faceDetector):
    print(imgPath)
    try:
        img = cv2.imread(imgPath)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = faceDetector(img, 1)
    except Exception as err:
        print(err)
        return True

    return True if len(dets) != 1 else False

def deleteFile(fileName):
    os.remove(fileName)

def isImage(fileName):
    return fileName.endswith('jpg')

# 这才是主函数
def swapForEverySingleFile(folder, isDebug=False):
    # 记录当前遍历的是第几个目录
    folderCount = 1
    for innerFolder in os.listdir(folder):
        folderName = os.path.join(folder, innerFolder)

        # 不是文件夹的时候删掉该文件
        if not os.path.isdir(folderName):
            #deleteFile(folderName)
            continue

        if  isDebug:
            print("current folder is %s" % (folderName))

        #继续对子文件夹进行遍历，这回里面全都是文件，而不是文件夹
        count = 1
        for image in os.listdir(os.path.join(folder, innerFolder)):
            fileName = os.path.join(folder, innerFolder, image)
            #判断原来的文件名是否有数字，有的话给一个随机的字符串，否则则为数字
            if hasNumbers(fileName):
                newFileName = os.path.join(folder, innerFolder, randomChars() + '.jpg')
            else:
                newFileName = os.path.join(folder, innerFolder, str(count) + '.jpg')

            os.chmod(fileName, stat.S_IWRITE)
            #print(judgeFaceNum(fileName, detector))
            #当当前文件不是jpg文件和当前图片内有多张人脸的时候就删除掉
            if not isImage(fileName):# or judgeFaceNum(fileName, detector):
                deleteFile(fileName)
            else:
                renameFile(fileName, newFileName)
                count += 1

        # 如果是文件夹的话，顺便改个名字
        if hasNumbers(folderName):
            renameFile(oldFileName=folderName, newFileName=os.path.join(folder, randomChars(4)))
        else:
            renameFile(oldFileName=folderName, newFileName=os.path.join(folder, str(folderCount)))
            folderCount += 1

def makeDirNameToNumber(folder, isDebug=False):
    folderCount = 1
    randomString = None
    for innerFolder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, innerFolder)):
            randomString = randomChars(10)
            newFileName = str(folderCount)
            temFileName = randomString
            renameFile(os.path.join(folder, innerFolder), os.path.join(folder, temFileName))
            renameFile(os.path.join(folder, temFileName), os.path.join(folder, newFileName))
            folderCount += 1

            if isDebug:
                print("Changine the %s to %s" % (innerFolder, newFileName))



if __name__ == '__main__':
    folder = '../image/train'
    swapForEverySingleFile(folder, True)