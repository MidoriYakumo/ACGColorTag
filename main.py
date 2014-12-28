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
import h5py, pickle

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

K = 8
ATTEMPTS = 10
MAX_ITER = 10
EPSILON = 1.0

CIR_RADIUS = SIZE_PREVIEW*4//(10*K)
CIR_MARGIN = SIZE_PREVIEW*1//(10*K)
CIR_CANVAS_SCALE = np.array(2)

CIR_BORDER_LIGHT = (222, 222, 222)
CIR_BORDER_DARK = (88, 88, 88)

tuning = True
comparison = True

#imported_pylab = False
fig = None

FOCUS_MODE = 'raw'
#FOCUS_MODE = 'weight-'
#FOCUS_MODE = 'weight+'

#os.popen('rm cache.hdf5')

HDF_REL_PATH = "/%(K)s/%(FOCUS_MODE)s/" % locals()

def better_imshow(title, img, reuse = True, block = False):
#	if title is None: title = "%sx%s" % img.shape[:2]
	if np.any(np.frombuffer(title.encode(), dtype = np.uint8)>=0x80):
		global fig
		from pylab import figure, subplot, imshow, show
		if not reuse or fig is None:
			fig = figure(figsize = (img.shape[1]/96, (img.shape[0]+30)/96))
#		plt = fig.add_subplot(111, title = title)
#		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		subplot(111, title = title)
		imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		show(block=block)

	else:
		cv2.imshow(title, img)
		imageDisplayed()

def imageDisplayed(delay = 1):
	c = cv2.waitKey(delay)
	if c>0:
		print("[Key pressed]", c)
		if c in [27, 113]: cv2.destroyAllWindows()

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
#	TOP_RATIO = 0.6

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
#			mask = (sobel/sobel.max())**2
	else:
		sobel_1d = sobel.flatten()
		mask = sobel>np.sort(sobel_1d)[TOP_RATIO*len(sobel_1d)]

		mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
							np.ones(F_SHAPE, dtype = np.uint8))

	if kwargd.get('rtn_img'):
		return (img * mask[:,:,np.newaxis]).astype(np.uint8)
	else:
		if kwargd.get('weight_mul'):
			mask *= img.shape[0] * img.shape[1] / np.sum(mask)
			mask += 0.5
			return np.repeat(img.reshape(-1,3), mask.flatten().astype(np.uint8), axis=0)
		if kwargd.get('rtn_weight'):
			return mask
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

def color_extractor(fn, mode):
	img = cv2.imread(fn)
	thumb = thumbnail(img, SIZE_THUMB, cv2.INTER_LINEAR)

	if FOCUS_MODE == 'remove':
		thumb = focus(thumb)

	if FOCUS_MODE == 'weight-':
		thumb = focus(thumb, rtn_img = True, rtn_weight = True)

	if FOCUS_MODE == 'weight+':
		thumb = focus(thumb, rtn_weight = True, weight_mul = True)

	colors = color_quantization(thumb)

	preview = None
	if mode == 'palette':
		preview = np.zeros(((CIR_RADIUS + CIR_MARGIN)*2, SIZE_PREVIEW,3), dtype = np.uint8)
	if mode == 'overlay':
		preview = thumbnail(img, SIZE_PREVIEW)

	if preview is not None:
		draw_palette(preview, colors)
#		cv2.imshow(str(fn.encode())+' damn no unicode support', preview)
		better_imshow(fn, preview, reuse = True)
		imageDisplayed()

	return colors

def char_extractor(fn):
	return [fn.split('.')[0]] # CHARACTER.X.jpg

def cache(key, miss_func = None, mode='r', fn = 'cache.hdf5', is_obj = False):
	if not os.path.isfile(fn):
		hdf = h5py.File(fn, "w")
		hdf.close()

	hdf = h5py.File(fn, "r+")
	# TODO check version
	if key in hdf:
		print('[Cache hit] ' + key)
		ret = np.array(hdf[key])
		if is_obj:
			ret = pickle.loads(ret.tobytes())
	else:
		ret = miss_func()
		if type(ret) is not np.ndarray:
			print('[Cache] ret of miss_func('+key+')is not an array')
		if 'w' in mode:
			if is_obj:
				data = np.frombuffer(pickle.dumps(ret), dtype = np.uint8)
			else:
				data = ret
			hdf.create_dataset(key, data=data)
	hdf.close()
	return ret

class illust_ds():
	def __init__(self):
		self.X, self.y= [], []
		self.c2y, self.y2c = {}, {}
		self.cntr = 0
		self.cnt = {}
		self.size = 0

	def append(self, features, characters):
		if len(characters) > 1:
			print("[Dataset] Only 1 character supported now.")
			return 1
		self.X.append(features)
		char = characters[0]
		if char not in self.c2y:
			self.y2c[self.cntr] = char
			self.c2y[char] = self.cntr
			self.cnt[char] = 0
			self.cntr += 1
		self.y.append(self.c2y[char])
		self.cnt[char] += 1
		self.size += 1

	def __repr__(self):
		return "illust_ds(size=%(size)s, cnt=%(cnt)s)" % self.__dict__

	def genIndices(self, cnt = 30):
		self.X = np.array(self.X)
		self.y = np.array(self.y)
		indice = list(range(len(self.X)))
		self.indices = []
		for i in range(cnt):
			np.random.shuffle(indice)
			self.indices.append(indice.copy())

def prepare_dataset(dsDir = 'dataset'):
	dsDir += '/'
	fileList = list(os.listdir(dsDir))
	total = len(fileList)
	print("[Info] Extracting features from {} files.".format(total))
	dataset = illust_ds()
	for i, fn in enumerate(fileList):
#		try:
			if fn.endswith('jpg') or fn.endswith('png'):
				print('[Info] {:4}/{:4}:{}'.format(i, total, fn))
				colors = cache('raw/'+HDF_REL_PATH+md5(dsDir+fn),
					lambda :color_extractor(dsDir + fn, 'palette').reshape(-1),
					'w')
				dataset.append(colors, char_extractor(fn))
#		except Exception as e:
#			print(fn, e)

	return dataset


def md5(filename):
	return os.popen("md5sum '"+filename+"'").read().split()[0]

if __name__ == '__main__':
	print('[Image feature] Mode:'+FOCUS_MODE)
	dataset = cache('/classifier/dataset'+HDF_REL_PATH+'image', lambda :prepare_dataset(), 'w', is_obj=True)
#	dataset = prepare_dataset()
	print("[Classifier] Dataset y={}, count = {}".format(dataset.c2y, dataset.cnt))
	dataset.genIndices()

	tuning = False
	if tuning:
		sMax = 0
#		for p in np.linspace(-20.0, 20.0, 15):
#		for p in np.linspace(1e-4, 2.0, 15):
		for p in range(4,6):
			param = dict(kernel = 'poly', degree = p, gamma = 1.8)
			clf = SVC(**param)

			s = 0
			for r in dataset.indices:
				X_c, y_c = dataset.X[r], dataset.y[r]
				sCnt = int(0.9 * dataset.size)
				clf.fit(X_c[:sCnt], y_c[:sCnt])
				s += clf.score(X_c[sCnt:],y_c[sCnt:])
			s /= len(dataset.indices)
			print("[Classifier] Cross_valid score = {:2},\t param = {}".format(s, param))
			if s>sMax: clfBest, sMax = clf, s
		print("[Classifier] ", clfBest.get_params())
		classifiers = [clfBest]
	else:
		classifiers = [
			SVC(kernel = 'poly', degree = 4, gamma = 1.8), # fair
			KNeighborsClassifier(n_neighbors = 25, weights = 'distance'), # fair+
			LDA(), # good
#			NuSVC(kernel = 'linear', nu = p, degree = 5), #suck
			MultinomialNB(), # fair
			GaussianNB(), # fair
#			BernoulliNB(), # suck
#			QDA(), # bug
			AdaBoostClassifier(), # bad
			BaggingClassifier(n_estimators = 45), # fair+
			DecisionTreeClassifier(), # bad
#			ExtraTreeClassifier(), # bad
			RandomForestClassifier(), # bad
#			RadiusNeighborsClassifier(), # suck eee
			LabelPropagation(kernel = 'knn'), #suck
#			LabelSpreading(kernel = 'knn'), #suck
			GradientBoostingClassifier(n_estimators = 31) # fair+ slow
			]
#		comparison = False
		if not comparison:
			classifiers = classifiers[:1]
	for clf in classifiers:
		clfName = str(clf).split('(')[0]
		s = 0
		for r_c in dataset.indices:
			X_c, y_c = dataset.X[r_c], dataset.y[r_c]
			sCnt = int(0.9 * dataset.size)
			clf.fit(X_c[:sCnt], y_c[:sCnt])
			s += clf.score(X_c[sCnt:],y_c[sCnt:])

		s /= len(dataset.indices)
		print("[Classifier] {} cross_valid score:{:3}".format(clfName, s))

		clf.fit(dataset.X, dataset.y)
		print("[Classifier] {} Training score:{}".format(clfName, clf.score(dataset.X, dataset.y)))

		if len(sys.argv) > 1:
			if len(sys.argv) > 2:
				files = sys.argv[1:]
			else : # Expand wildcard
				files = glob.glob(sys.argv[1])

			for i, fn in enumerate(files):
				img = cv2.imread(fn)
				thumb = thumbnail(img, SIZE_THUMB, cv2.INTER_LINEAR)

				if FOCUS_MODE == 'remove':
					thumb = focus(thumb)

				if FOCUS_MODE == 'weight-':
					thumb = focus(thumb, rtn_img = True, rtn_weight = True)

				if FOCUS_MODE == 'weight+':
					thumb = focus(thumb, rtn_weight = True, weight_mul = True)

				colors = color_quantization(thumb)

				y_test = clf.predict(colors.reshape(-1))[0]
				char = dataset.y2c[y_test]
				print("[Classifier] {}: May {} in {}.".format(clfName, char, fn))

				preview = thumbnail(img, SIZE_PREVIEW)
				preview = focus(preview, rtn_img = True, rtn_weight = True)
				draw_palette(preview, colors)
#				better_imshow(clfName+':'+char, preview)
#				better_imshow(clfName+':'+char, preview, reuse = False)
#				imageDisplayed()
				cv2.imwrite('result/'+fn.split('/')[-1]+'-'+
					clfName+'-'+char+'-'+FOCUS_MODE+'.jpg', preview)