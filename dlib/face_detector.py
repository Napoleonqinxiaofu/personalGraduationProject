#!/usr/bin/python
import sys

import dlib
import os
from PIL import Image

import cv2
import numpy


detector = dlib.get_frontal_face_detector()

folder = '../image/FERET/'

for file in os.listdir(folder):
    imagePath = os.path.join(folder, file)
    if not imagePath.endswith('tif'):
        continue
    print("Processing file: {}".format(imagePath))
    try:
        # img = cv2.imread(imagePath)
        img = numpy.array(Image.open(imagePath).convert('L'))
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = detector(img, 2)
    except Exception as err:
        print( err )
        continue

    print(len(dets))
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0) & 0XFF
cv2.destroyAllWindows()