#_*_coding:utf-8_*_
import dlib
import cv2
import numpy
import os
import sys
sys.path.append('../tool')
import imtool

class Face:
	def __init__(self, shape_predict_path='../dlib/shape_predictor_68_face_landmarks.dat',
		recognition_model_path='../dlib/dlib_face_recognition_resnet_model_v1.dat'):
	
		self.detector = dlib.get_frontal_face_detector()
		self.shape_detector = dlib.shape_predictor(shape_predict_path)
		self.faceRecognition_model = dlib.face_recognition_model_v1(recognition_model_path)

		self.face_rects = []

		self.trainEncodes = []
		self.trainLable = []

	def _encode(self, img):
		self.face_rects = []

		encodes = []

		rects = self.detector(img, 1)
		if len(rects) == 0:
			return None

		for rect in rects:
			self.face_rects.append((rect.left(), rect.top(), rect.right(), rect.bottom()))

			#68个landmark点
			shape = self.shape_detector(img, rect)
			#开始计算128维特征向量
			face_descriptor = self.faceRecognition_model.compute_face_descriptor(img, shape)
			encodes.extend([list(face_descriptor)])
		return numpy.array(encodes)

	def _distance(self, testEncode):
		distance = []
		for encode in self.trainEncodes:
			distance.append(numpy.linalg.norm(encode - testEncode))
		return numpy.array(distance)

	def drawRect(self, img, rect):
		cv2.rectangle(img, rect[:2], rect[2:4], (0, 255, 0), 2)

	def putText(self, img, text, rect):
		cv2.putText(img, text, rect[:2], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255))

	def turnToScore(self, score):
		"""
		将编码计算之后得到的结果转换成百分制，阈值是0.6
		score ： 需要转换的数值（一般情况下是通过_distance函数获取到的分数）
		return ： 百分制分数
		"""
		percentScore = 100 if score <= 0.2 else (score-0.2)/0.4 * 100
		return percentScore


	def train(self, trainFloder, isDebug=False):
		"""
		训练阶段，将所有的训练图像放置在trainFolder目录下，由该函数进行读取，当isDebug为True的时候会在每处理一张图片时在控制台显示该图片的信息。
		"""
		self.trainEncodes = []
		self.trainLable = []
		for image in os.listdir(trainFloder):
			if not image.endswith('jpg'):
				continue

			if isDebug:
				print(image)

			img = cv2.imread(os.path.join(trainFloder, image))
			encodes = self._encode(img)

			if encodes is None:
				continue
			#每一张图片中可能有两张人脸，谨慎起见，但是在训练的时候希望尽量使用每张图片就一张人脸的图片较好
			for en in encodes:
				self.trainEncodes.extend([en])
				self.trainLable.append(image.split('.')[0])

		#都将训练编码以及标签转换成numpy数组，numpy数组比原生数组更好操作一些
		self.trainEncodes = numpy.array(self.trainEncodes)
		self.trainLable = numpy.array(self.trainLable)


	def predict(self, testImage, k=3, thresh=0.6):
		"""
		预测的函数，第一个参数是需要预测的图像数据——RGB颜色的
					第二个参数表示需要在训练数据中获取多少个结果（默认是3个），有的时候预测到的结果比较多，传递这个参数可以控制获取的结果。
					第三个参数表示阈值，默认是0.6，在两张人脸编码之间的距离小于该阈值的时候才会被认为是同一张人脸

					return : 数组，每一个数组包含的信息有：
						1 : 当前人脸在测试图像中的位置
						2 : 当前人脸在训练集中进行比较得到的最像的几张图像的分数
						3 : 第二个参数所对应的所得的分数对应的训练图像标签
		"""
		#获取当前图片的编码向量，有可能有多张人脸，需要注意
		testEncode = self._encode(testImage)

		result = []

		for i, encode in enumerate(testEncode):
			#获取当前人脸编码与所有的训练图像编码之间的距离
			results = self._distance(encode)
			sortIndex = numpy.argsort(results)

			#获取训练图像中最有可能与测试图片是同一张脸的值
			scores = numpy.sort(results[numpy.where(results<thresh)])

			k = k if len(scores) >= k else len(scores)
			
			result.append([self.face_rects[i], scores[:k], self.trainLable[sortIndex[:k]] ])

		return result



if __name__ == '__main__':
	def path(lableName):
		return '../image/xiaoTrain/' + str(lableName) + '.jpg'

	img = '../image/xiaoTest/10.jpg'

	img = cv2.imread(img)
	img = imtool.resize(img, width=400)

	faceReconize = Face()

	faceReconize.train('../image/xiaoTrain', True)
	score = faceReconize.predict(img)
	
	for s in score:
		faceReconize.drawRect(img, s[0])

		faceReconize.putText(img, 'Who stole the panties?', s[0][:2])

		predictImage = None
		count = 1
		for i in s[2]:
			image = imtool.resize(cv2.imread(path(i)), width=400)
			faceReconize.putText(image, 'Maybe that is him!', (30, 30))
			# cv2.imwrite('test' + str(count) + '.jpg', image)
			count += 1
			cv2.imshow('test' + str(count) + '.jpg', image)


			# if predictImage is None:
			# 	predictImage = image
			# else:
			# 	predictImage = numpy.hstack((predictImage, image))

	cv2.imshow('img', img)
	# cv2.imwrite('trainImage.jpg', img)
	# cv2.imshow('img2', predictImage)
	cv2.waitKey(0) & 0XFF
	cv2.destroyAllWindows()