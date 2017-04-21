#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/26-16:56
description : 
"""

import numpy as np

def _face_distance(faces, face_to_compare):
    """
    Given a list of face encodings, compared them to a known face encoding and get a euclidean distance
    for each comparison face.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A list with the distance for each face in the same order as the 'faces' array
    """
    return np.linalg.norm(faces - face_to_compare, axis=1)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(_face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def find_similar_face(known_face_encodings, face_encoding_to_check, k=3, tolerance=0.6):
	"""
	从多张图片中寻找与face_encoding_to_check最相近的图片
	:param known_face_encodings:
	:param face_encoding_to_check:
	:param k:获取最邻近的几个图片
	:param tolerance:
	:return:
	"""
	#获取最初始的评分
	raw_results = _face_distance(known_face_encodings, face_encoding_to_check)
	sortIndex = np.argsort(raw_results)
    # if isDebug:
    #     print("the k score is")
    #     print([raw_results[sortIndex[idx]] for idx in range(k)])
	return sortIndex[:k]