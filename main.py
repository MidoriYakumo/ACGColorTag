# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 13:46:34 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""
import cv2
import numpy as np
import os, sys, time, glob
import h5py

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.semi_supervised import LabelPropagation, LabelSpreading




__version__ = '0.0.1'

SIZE_THUMB = 256
SIZE_PREVIEW = 512

K = 6
ATTEMPTS = 10
MAX_ITER = 10
EPSILON = 1.0

CIR_RADIUS = SIZE_PREVIEW*4//(10*K)
CIR_MARGIN = SIZE_PREVIEW*1//(10*K)
CIR_CANVAS_SCALE = np.array(2)

CIR_BORDER_LIGHT = (222, 222, 222)
CIR_BORDER_DARK = (88, 88, 88)



def thumbnail(img, dMax, resample = cv2.INTER_LINEAR):

	# Generate thumbnail and preview
	height, width = img.shape[:2]
	downscale = min(width, height)/dMax
	return cv2.resize(img, (int(width/downscale), int(height/downscale)), resample)

def focus(img, **kwargs):
	'''
	focus on character for BGR image
	'''

	kwargd = dict()
	kwargd.update(kwargs)

#	F_SHAPE = (9, 9)
	F_SHAPE = (img.shape[0]//3, img.shape[1]//3)
	TOP_RATIO = 0.7

	sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5).astype(np.int32)
	sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5).astype(np.int32)
	sobel = np.sum(sobelx**2 + sobely**2, axis = 2)**0.5
	sobel = cv2.blur(sobel, F_SHAPE)

	if kwargd.get('rtn_weight'):
		weight_func = kwargd.get('weight_func')
		if weight_func:
			mask = weight_func(sobel)
		else:
			mask = sobel/sobel.max()
	else:
		sobel_1d = sobel.flatten()
		mask = sobel>np.sort(sobel_1d)[TOP_RATIO*len(sobel_1d)]

		mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
							np.ones(F_SHAPE, dtype = np.uint8))

	if kwargd.get('rtn_img'):
		return (img * mask[:,:,np.newaxis]).astype(np.uint8)
	else:
		if kwargd.get('rtn_weight'):
			return mask
		else:
			return img[np.where(mask)]

def color_quantization(img, **kwargs):
	'''
	color quantization for BGR image
	'''

	kwargd = dict(sort = 'L')
	kwargd.update(kwargs)

	termcrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITER, EPSILON)
	_, _, centroids = cv2.kmeans(img.reshape(-1,3).astype(np.float32), K, None, termcrit, ATTEMPTS, cv2.KMEANS_PP_CENTERS)

	centroids = centroids.astype(np.uint8)

	c_rank = list(range(K))
	c_hls = cv2.cvtColor(centroids[:,np.newaxis], cv2.COLOR_BGR2HLS)[:,0,:]
	sort = kwargd['sort']
	if sort == 'H':
		c_rank = sorted(c_rank, key=lambda n:c_hls[n,0])
	if sort == 'L':
		c_rank = sorted(c_rank, key=lambda n:c_hls[n,1])

	ccentroids = centroids[c_rank]

	return ccentroids

def draw_palette(img, colors):
	for i, c in enumerate(colors):
		c = [int(n) for n in c]
		cv2.circle(img, ((CIR_MARGIN + CIR_RADIUS)*(2*i+1), CIR_MARGIN + CIR_RADIUS), CIR_RADIUS, c, -1, cv2.LINE_AA)
		cv2.circle(img, ((CIR_MARGIN + CIR_RADIUS)*(2*i+1), CIR_MARGIN + CIR_RADIUS) , CIR_RADIUS - 2, CIR_BORDER_LIGHT, 3, cv2.LINE_AA)
		cv2.circle(img, ((CIR_MARGIN + CIR_RADIUS)*(2*i+1), CIR_MARGIN + CIR_RADIUS) , CIR_RADIUS, CIR_BORDER_DARK, 1, cv2.LINE_AA)

def imageDisplayed(delay = 1):
	c = cv2.waitKey(delay)
	if c>0:
		print("[Key pressed]", c)
		if c in [27, 113]: cv2.destroyAllWindows()

def color_extractor(fn, mode):
	img = cv2.imread(fn)
	thumb = thumbnail(img, SIZE_THUMB, cv2.INTER_LINEAR)

	body = focus(thumb)
	colors = color_quantization(body)

	preview = None
	if mode == 'palette':
		preview = np.zeros(((CIR_RADIUS + CIR_MARGIN)*2, SIZE_PREVIEW,3), dtype = np.uint8)
	if mode == 'overlay':
		preview = thumbnail(img, SIZE_PREVIEW)

	if preview is not None:
		draw_palette(preview, colors)
		cv2.imshow(str(fn.encode())+' damn no unicode support', preview)
		imageDisplayed()

	return colors

def char_extractor(fn):
	return [fn.split('.')[0]] # CHARACTER.X.jpg

def append_chars(chars):
	global y2c, c2y, cCnt, cCntr
	if len(chars) > 1:
		print("[Dataset] Only 1 character supported now.")
		return 0
	char = chars[0]
	if char not in c2y:
		y2c[cCntr] = char
		c2y[char] = cCntr
		cCnt[char] = 0
		cCntr += 1
	y.append(c2y[char])
	cCnt[char] += 1

def cache(key, miss_func = None, mode='r', fn = 'cache.hdf5'):
	if not os.path.isfile(fn):
		hdf = h5py.File(fn, "w")
		hdf.close()

	hdf = h5py.File(fn, "r+")
	# TODO check version
	if key in hdf:
		print('[Cache hit] ' + key)
		ret = np.array(hdf[key])
	else:
		ret = miss_func()
		if type(ret) is not np.ndarray:
			print('[Cache] ret of miss_func('+key+')is not an array')
		if 'w' in mode:
			hdf.create_dataset(key, data=ret)
	hdf.close()
	return ret

def md5(filename):
	return os.popen("md5sum '"+filename+"'").read().split()[0]


if __name__ == '__main__':
	X, y= [], []
	c2y, y2c = {}, {}
	cCntr = 0
	cCnt = {}

	dsDir = 'img/'
	fileList = list(os.listdir(dsDir))
	total = len(fileList)
	print("[Info] Extracting features from {} files.".format(total))
	for i, fn in enumerate(fileList):
#		try:
			if fn.endswith('jpg') or fn.endswith('png'):
				print('[Info] {:4}/{:4}:{}'.format(i, total, fn))
				colors = cache(md5(dsDir+fn), lambda :color_extractor(dsDir + fn, 'palette').reshape(-1), 'w')
				X.append(colors)
				append_chars(char_extractor(fn))
#		except Exception as e:
#			print(fn, e)

	print("[Classifier] Dataset y={}, count = {}".format(c2y, cCnt))

	imageDisplayed(0)

#
#if __name__ == '__main__':
#	X = np.array(X)
#	y = np.array(y)
#	r_c = list(range(len(X)))
#	r_cs = []
#	for i in range(30):
#		np.random.shuffle(r_c)
#		r_cs.append(r_c.copy())
#
##	tuning = False
#	if tuning:
#
#		sMax = 0
##		for p in np.linspace(-20.0, 20.0, 15):
##		for p in np.linspace(1e-4, 2.0, 15):
#		for p in range(4,6):
#			param = dict(kernel = 'poly', degree = p, gamma = 1.8)
#
#			clf = SVC(**param)
#
#			s = 0
#			for r_c in r_cs:
#				X_c, y_c = X[r_c], y[r_c]
#				sCnt = int(0.9 * len(X))
#				clf.fit(X_c[:sCnt], y_c[:sCnt])
#				s += clf.score(X_c[sCnt:],y_c[sCnt:])
#
#			s /= len(r_cs)
#			print("[Classifier] Cross_valid score = {:2},\t param = {}".format(s, param))
#
#			if s>sMax: clfBest, sMax = clf, s
#
#		print("[Classifier] ", clfBest.get_params())
#		classifiers = [clfBest]
#
#	else:
#		classifiers = [
#			SVC(kernel = 'poly', degree = 4, gamma = 1.8), # fair
##			NuSVC(kernel = 'linear', nu = p, degree = 5), #suck
#			MultinomialNB(), # fair
#			GaussianNB(), # fair
##			BernoulliNB(), # suck
#			LDA(), # good
##			QDA(), # bug
#			AdaBoostClassifier(), # bad
#			GradientBoostingClassifier(n_estimators = 31), # fair+ slow
#			BaggingClassifier(n_estimators = 45), # fair+
#			DecisionTreeClassifier(), # bad
##			ExtraTreeClassifier(), # bad
#			RandomForestClassifier(), # bad
##			RadiusNeighborsClassifier(), # suck eee
#			LabelPropagation(kernel = 'knn'), #suck
##			LabelSpreading(kernel = 'knn'), #suck
#			KNeighborsClassifier(n_neighbors = 25, weights = 'distance') # fair+
#			]
##		comparison = False
#		if not comparison:
#			classifiers = classifiers[:1]
#	for clf in classifiers:
#		clfName = str(clf).split('(')[0]
#		s = 0
#		for r_c in r_cs:
#			X_c, y_c = X[r_c], y[r_c]
#			sCnt = int(0.9 * len(X))
#			clf.fit(X_c[:sCnt], y_c[:sCnt])
#			s += clf.score(X_c[sCnt:],y_c[sCnt:])
#
#		s /= len(r_cs)
#		print("[Classifier] {} cross_valid score:{:3}".format(clfName, s))
#
#		clf.fit(X, y)
#		print("[Classifier] {} Training score:{}".format(clfName, clf.score(X,y)))
#
#		if len(sys.argv) > 1:
#			if len(sys.argv) > 2:
#				files = sys.argv[1:]
#			else : # Expand wildcard
#				files = glob.glob(sys.argv[1])
#
#			for i, fn in enumerate(files):
#				X_test = denominant_color(fn, show_img = None)
#				y_test = y2c[clf.predict(X_test.reshape(-1))[0]]
#				print("[Classifier] May {} in {}.".format(y_test, fn))
#
#				if disp:
#					text = y_test + '/' + clfName
#					img = denominant_color(fn, rtn_img = True,
#						show_img = 'overlay')
#					canvas = Image.new('RGBA', img.size)
#					painter = ImageDraw.Draw(canvas)
#					if os.path.isfile(FONT):
#						font = ImageFont.truetype(FONT, 36)
#					else:
#						font = ImageFont.load_default(36)
#					painter.text((CIR_MARGIN +2, CIR_RADIUS * 2 + CIR_MARGIN * 3 +2),
#						text, font=font,
#						fill = RGBA_SHADOW)
#					painter.text((CIR_MARGIN, CIR_RADIUS * 2 + CIR_MARGIN * 3),
#						text, font=font,
#						fill = tuple([int(c) for c in X_test[int(len(X_test)*2/3)]]))
#					Image.alpha_composite(img, canvas).show(title = fn)
#
#
#
#