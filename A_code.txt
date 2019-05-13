import cv2
import numpy as np
import matplotlib.pyplot as plt


class EdgeDetect(object):

	def __init__(self, path):
		self.image = cv2.imread(path)
		self.gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	def sobel(self):
		sobelX = cv2.convertScaleAbs(cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize = 3))
		sobelY = cv2.convertScaleAbs(cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize = 3))
		sobelXY_3 = cv2.convertScaleAbs(cv2.Sobel(self.gray, cv2.CV_64F, 1, 1, ksize = 3))
		sobelXY_5 = cv2.convertScaleAbs(cv2.Sobel(self.gray, cv2.CV_64F, 1, 1, ksize = 5))
		sobelXY_7 = cv2.convertScaleAbs(cv2.Sobel(self.gray, cv2.CV_64F, 1, 1, ksize = 7))
		names = ['Original', 'X--3x3', 'Y--3x3', 'XY--3x3', 'XY--5x5', 'XY--7x7']
		images = [self.gray, sobelX, sobelY, sobelXY_3, sobelXY_5, sobelXY_7]
		self.draw(names, images, 2, 3)

	def total_sobel(self):
		namelist = ['./test1.tif', './test2.png', './test3.jpg', './test4.bmp', 'test5.png', './test6.jpg']
		images = []
		for name in namelist:
			image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
			images.append(self.s(image))
		self.draw(['test' + str(i) for i in range(1, len(namelist) + 1)], images, 2, 3)

	def s(self, img):
		x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
		y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
		absX = cv2.convertScaleAbs(x)
		absY = cv2.convertScaleAbs(y)
		dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
		return dst

	def canny(self):
		edge = cv2.Canny(self.gray, 100, 200)
		names = ['Original', 'Canny--100--200']
		images = [self.gray, edge]
		self.draw(names, images, 1, 2)

	def hough(self, key):
		namelist = ['./test1.tif', './test2.png', './test3.jpg', './test4.bmp', 'test5.png', './test6.jpg']
		images = []
		if key == 'canny':
			for name in namelist:
				image = cv2.imread(name)
				gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
				edge = cv2.Canny(gray, 100, 200)
				lines = cv2.HoughLinesP(edge, 1, np.pi/180, 20)[:,0,:]
				for x1,y1,x2,y2 in lines: 
					cv2.line(image,(x1,y1),(x2,y2),(255, 0, 0),2)
				images.append(image)
		elif key == 'sobel':
			for name in namelist:
				image = cv2.imread(name)
				gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
				x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
				y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
				absX, absY = cv2.convertScaleAbs(x), cv2.convertScaleAbs(y)
				dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
				dst[dst >= 40], dst[dst < 40] = 255, 0
				lines = cv2.HoughLinesP(dst, 1, np.pi/180, 20)[:,0,:]
				for x1,y1,x2,y2 in lines: 
					cv2.line(image,(x1,y1),(x2,y2),(255, 0, 0),1)
				images.append(image)
		self.draw(['test' + str(i) for i in range(1, len(namelist) + 1)], images, 2, 3)

	def draw(self, names, images, row, col):
		plt.figure('SHOW')
		for i in range(len(names)):
			plt.subplot(row, col, i+1)
			plt.title(names[i])
			plt.imshow(images[i], cmap='gray')
		plt.show()


if __name__ == '__main__':
	path = './test1.tif'
	path = './test2.png'
	path = './test3.jpg'
	path = './test4.bmp'
	path = 'test5.png'
	path = './test6.jpg'
	e = EdgeDetect(path)
	#e.sobel()
	#e.total_sobel()
	#e.canny()
	#e.hough('canny')
	e.hough('sobel')
