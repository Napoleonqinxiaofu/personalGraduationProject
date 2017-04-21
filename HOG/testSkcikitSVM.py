#_*_coding:utf-8_*_
"""
author:NapoleonQin
email:qxfnapoleon@163.com
weChat:echopi31415927
date : 2017/3/22-10:06
description : 学习一下scikit-learn中的SVM
"""

from sklearn import svm

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #   decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
  #  max_iter=-1, probability=False, random_state=None, shrinking=True,
   # tol=0.001, verbose=False)
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
print(dec) # 4 classes
print(clf.predict([[2]]))