#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 21:31:22 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""

from PIL import Image, ImageDraw, ImageFont
from colorsys import rgb_to_hsv, rgb_to_hls

import numpy as np
from sklearn.cluster import KMeans

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

SIZE_THUMB = 128
SIZE_PREVIEW = 512

CIR_RADIUS = 30
CIR_MARGIN = 10
CIR_CANVAS_SCALE = np.array(2)

RGBA_SHADOW = (66,66,66,180)

FONT = os.environ['HOME'] + "/.fonts/Android/DroidSansFallback.ttf"
disp = True
#disp = False
tuning = True
comparison = True

from PIL.ImageShow import UnixViewer, register

class FEHViewer(UnixViewer):
	def get_command_ex(self, file, title=None, **options):
		command = executable = "feh"
		if title:
			command += " --title '%s'" % (title)
		return command, executable

if os.path.isfile("/usr/bin/feh"):
	register(FEHViewer, -1)
	os.popen('killall feh')

K = 8
#KC = 3
#LOWSAT = 50

def denominant_color(filename, **kwargs):

	kwargsd = dict(show_img = 'colors', cache = 'cache.hdf5', cache_mode = 'r', vect_sort = 'L')
	kwargsd.update(kwargs)

	cRGB = None

	cache_mode = kwargsd.get('cache_mode')
	cache = kwargsd.get('cache')
	if cache:
		md5=os.popen("md5sum '"+filename+"'").read().split()[0]
		if os.path.isfile(cache):
			hdf = h5py.File(cache, "r+")
			# TODO check version
			if md5 in hdf:
				cRGB = np.array(hdf[md5])
				print('[Cache hit] ' + filename)
		elif 'w' in cache_mode:
			hdf = h5py.File(cache, "w")

	def putCircle(xy, color, **kwargs):
		r = CIR_RADIUS
		xy = np.array(xy)
		xy = np.array([xy - r, xy + r]).reshape(-1)
		color = [int(c) for c in color]
		color.append(255)
		painter.ellipse(tuple((xy+2) * CIR_CANVAS_SCALE), fill = RGBA_SHADOW)
		painter.ellipse(tuple(xy * CIR_CANVAS_SCALE), fill = tuple(color))

	# Generate thumbnail and preview
	img = Image.open(filename)
	width, height = img.size
	downscale = min(img.size)/SIZE_THUMB
#	thumb = img.resize((int(width/downscale), int(height/downscale)), Image.NEAREST)
	thumb = img.convert('RGB').resize((int(width/downscale), int(height/downscale)), Image.BILINEAR)

	# Draw circles onn canvas
	show_img = kwargsd.get('show_img')
	if show_img == 'colors':
		cir_box = (CIR_MARGIN + CIR_RADIUS) * 2
		xx, yy = (cir_box * K, cir_box) * CIR_CANVAS_SCALE
		canvas = Image.new('RGBA', (xx,yy))
		painter = ImageDraw.Draw(canvas)
		painter.rectangle((0, 0, xx, yy), fill='white')
	else:
		downscale = min(img.size)/SIZE_PREVIEW
		preview = img.resize((int(width/downscale), int(height/downscale)), Image.ANTIALIAS).convert('RGBA')
		canvas = Image.new('RGBA', preview.size * CIR_CANVAS_SCALE)
		painter = ImageDraw.Draw(canvas)


	if cRGB is None:
		# KMeans clustering
		clt = KMeans(K, n_jobs = -1,
	#		max_iter = 200, n_init = 8,
			init = "k-means++")
		clt.fit(np.array(thumb).reshape((-1,3)))

		# Sort colors into a vector
		rRank = list(range(K))
		cRGB = clt.cluster_centers_
		cHLS = np.array([rgb_to_hls(*c) for c in cRGB])

		vect_sort = kwargsd.get('vect_sort')
		if vect_sort == 'H': #	#Sort with Hue
			rRank = sorted(rRank, key=lambda n:cHLS[n,0])
		if vect_sort == 'L': # Sort with Luminance
			rRank = sorted(rRank, key=lambda n:cHLS[n,1])

	#	#Make it best cut on Hue
	#	cDL = cHLS[rRank][1:,0] - cHLS[rRank][:-1,0]
	##	print(1 + cHLS[rRank][0,0] - cHLS[rRank][-1,0], cDL)
	#	if cDL.max() > 1 + cHLS[rRank][0,0] - cHLS[rRank][-1,0]:
	#		rRank = rRank[cDL.argmax()+1:] + rRank[:cDL.argmax()+1]

		#Pass to output
		cRGB = cRGB[rRank]
	else:
		kwargsd['show_img'] = None

	if cache:
		if 'w' in cache_mode and md5 not in hdf:
			hdf.create_dataset(md5, data=cRGB)
		hdf.close()

	for i, color in enumerate(cRGB):
		putCircle(((2*i+1)*(CIR_MARGIN + CIR_RADIUS), (CIR_RADIUS + CIR_MARGIN)), color)

	if show_img == 'colors':
		preview = canvas.resize((cir_box * K, cir_box), Image.ANTIALIAS)

	if show_img == 'overlay':
		preview = Image.alpha_composite(preview, canvas.resize(preview.size, Image.ANTIALIAS))

	if kwargsd.get('rtn_img'): return preview
	if kwargsd.get('show_img'): preview.show(title = filename)

	return cRGB

if __name__ == '__main__':

	X = []
	y = []
	c2y = {}
	y2c = {}
	cCntr = 0
	cCnt = {}
	dsDir = 'img/'
	fileList = list(os.listdir(dsDir))
	total = len(fileList)
	print("[Info] Extracting features from {} files.".format(total))
	for i, fn in enumerate(fileList):
#		try:
			if fn.endswith('jpg') or fn.endswith('png'):
#				print('{:4}/{:4}:{}'.format(i, total, fn))
				X.append(denominant_color(dsDir + fn, cache_mode = 'w').reshape(-1))
				chars = fn.split('.')[0] # CHARACTER.X.jpg
				if chars not in c2y:
					y2c[cCntr] = chars
					c2y[chars] = cCntr
					cCnt[chars] = 0
					cCntr += 1
				y.append(c2y[chars])
				cCnt[chars] += 1
#		except Exception as e:
#			print(fn, e)

	print("[Classifier] Dataset y={}, count = {}".format(c2y, cCnt))

	X = np.array(X)
	y = np.array(y)
	r_c = list(range(len(X)))
	r_cs = []
	for i in range(30):
		np.random.shuffle(r_c)
		r_cs.append(r_c.copy())

	tuning = False
	if tuning:

		sMax = 0
#		for p in np.linspace(-20.0, 20.0, 15):
#		for p in np.linspace(1e-4, 0.20, 15):
		for p in range(2, 20):
			param = dict()

			clf = KNeighborsClassifier(**param)

			s = 0
			for r_c in r_cs:
				X_c, y_c = X[r_c], y[r_c]
				sCnt = int(0.9 * len(X))
				clf.fit(X_c[:sCnt], y_c[:sCnt])
				s += clf.score(X_c[sCnt:],y_c[sCnt:])

			print("[Classifier] Cross_valid score = {:2},\t param = {}".format(s, param))

			if s>sMax: clfBest, sMax = clf, s

		clf = clfBest
		print("[Classifier] ", clf.get_params())

	else:
		classifiers = [
			SVC(kernel = 'poly', C = 1., degree = 5, gamma = 0.8, tol = 1.0), # fair
#			NuSVC(kernel = 'linear', nu = p, degree = 5), #suck
			MultinomialNB(), # fair
			GaussianNB(), # fair
#			BernoulliNB(), # suck
			LDA(), # good
#			QDA(), # bug
			AdaBoostClassifier(), # bad
			GradientBoostingClassifier(), # fair+ slow
			BaggingClassifier(n_estimators = 45), # fair+
			DecisionTreeClassifier(), # bad
#			ExtraTreeClassifier(), # bad
			RandomForestClassifier(), # bad
#			RadiusNeighborsClassifier(), # suck eee
			LabelPropagation(kernel = 'knn'), #suck
#			LabelSpreading(kernel = 'knn'), #suck
			KNeighborsClassifier(n_neighbors = 5, weights = 'distance') # fair
			]
#		comparison = False
		if not comparison:
			classifiers = classifiers[:1]
		for clf in classifiers:
			clfName = str(clf).split('(')[0]
			s = 0
			for r_c in r_cs:
				X_c, y_c = X[r_c], y[r_c]
				sCnt = int(0.9 * len(X))
				clf.fit(X_c[:sCnt], y_c[:sCnt])
				s += clf.score(X_c[sCnt:],y_c[sCnt:])

			print("[Classifier] {} cross_valid score:{:3}".format(clfName, s/len(r_cs)))

			clf.fit(X, y)
			print("[Classifier] {} Training score:{}".format(clfName, clf.score(X,y)))

			if len(sys.argv) > 1:
				if len(sys.argv) > 2:
					files = sys.argv[1:]
				else : # Expand wildcard
					files = glob.glob(sys.argv[1])

				for i, fn in enumerate(files):
					X_test = denominant_color(fn, show_img = None)
					y_test = y2c[clf.predict(X_test.reshape(-1))[0]]
					print("[Classifier] May {} in {}.".format(y_test, fn))

					if disp:
						text = y_test + '/' + clfName
						img = denominant_color(fn, rtn_img = True,
							show_img = 'overlay')
						canvas = Image.new('RGBA', img.size)
						painter = ImageDraw.Draw(canvas)
						if os.path.isfile(FONT):
							font = ImageFont.truetype(FONT, 36)
						else:
							font = ImageFont.load_default(36)
						painter.text((CIR_MARGIN +2, CIR_RADIUS * 2 + CIR_MARGIN * 3 +2),
							text, font=font,
							fill = RGBA_SHADOW)
						painter.text((CIR_MARGIN, CIR_RADIUS * 2 + CIR_MARGIN * 3),
							text, font=font,
							fill = tuple([int(c) for c in X_test[int(len(X_test)*2/3)]]))
						Image.alpha_composite(img, canvas).show(title = fn)
