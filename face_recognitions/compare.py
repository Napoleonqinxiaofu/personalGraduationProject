#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/26-11:14
description : 对比两张照片
"""
import face_recognition as face
import cv2
import numpy

import os
import dlib

import api

known = '../image/'

train = []
count = 0
for image in os.listdir(known):

	image = face.load_image_file(os.path.join(known, image))
	face_locations = face.face_locations(image)
	if len(face_locations) < 1:
		continue
	count += 1
	if count > 6:
		break

	train.extend([face.face_encodings(image)[0]])
results = api.find_similar_face(train, train[-1])
print(results)

# 	top, right, bottom, left = face_locations[0]
# 	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
# 	cv2.imshow('img', image)
# 	cv2.waitKey(300) & 0XFF
# cv2.destroyAllWindows()