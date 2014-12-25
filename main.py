#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 21:31:22 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""

from PIL import Image, ImageDraw
from colorsys import rgb_to_hsv, rgb_to_hls
import numpy as np
from sklearn.cluster import KMeans
import os, sys, time
import h5py


from sklearn import svm

__version__ = '0.0.1'

cSize = 30
cMargin = 10

def denominant_color(filename, showImg = True, cache = 'cache.hdf5'):
	if cache:
		md5=os.popen('md5sum '+filename).read().split()[0]
		if not os.path.isfile(cache):
			hdf = h5py.File(cache, "w")
			hdf.close()

		# TODO check version

		hdf = h5py.File(cache, "r+")
		if md5 in hdf:
			cRGB = np.array(hdf[md5])
			hdf.close()
			return cRGB


	def putCircle(xy, color):
		r = cSize
		xy = np.array(xy)
		xy = np.array([xy - r, xy + r]).reshape(-1)
		color = [int(c) for c in color]
		color.append(255)
		painter.ellipse(tuple((xy+2) * cScale), fill = (99,99,99,180))
		painter.ellipse(tuple(xy * cScale), fill = tuple(color))

	img = Image.open(filename).convert('RGBA')
	width, height = img.size
	downscale = min(img.size)/128
	thumb = img.resize((int(width/downscale), int(height/downscale))).convert('RGB')

	K = 16
	clt = KMeans(K, max_iter = 200, n_init = 8, n_jobs = -1,
		init = "k-means++")
	clt.fit(np.array(thumb).reshape((-1,3)))


	cScale = np.array(2)
	canvas = Image.new('RGBA', img.size * cScale)
	painter = ImageDraw.Draw(canvas)

	cCnt = np.zeros(K)
	for c in clt.labels_: cCnt[c] += 1
	cRank = sorted(range(K), key=lambda n:cCnt[n], reverse=True)
	cRGB = clt.cluster_centers_[cRank[:int(K/2)]]
#	cHSX = [rgb_to_hls(*c) for c in cRGB]
	cHSX = [rgb_to_hsv(*c) for c in cRGB]
	cRank = sorted(range(int(K/2)), key=lambda n:cHSX[n][1], reverse=True)
#	cRank = sorted(range(int(K/2)), key=lambda n:cHSX[n][2])
	cRGB = cRGB[cRank]
	cHSX = np.array(cHSX)[cRank]

	cRGB = cRGB.astype(np.uint8)

	for i, color in enumerate(cRGB):
		putCircle((cSize + cMargin, (2*i+1)*(cMargin + cSize)), color)

	img = Image.alpha_composite(img, canvas.resize(img.size, Image.ANTIALIAS))

	if showImg: img.show()
	if cache:
		hdf.create_dataset(md5, data=cRGB)
		hdf.close()
	return cRGB

if __name__ == '__main__':
	X = []
	y = []
	c2y = {}
	y2c = {}
	cCnt = 0
	counter = {}
	imgdir = 'img/'
	for fn in os.listdir(imgdir):
		if fn.endswith('jpg') or fn.endswith('png'):
			X.append(denominant_color(imgdir + fn).reshape(-1))
			chars = fn.split('.')[0]
			if chars not in c2y:
				y2c[cCnt] = chars
				c2y[chars] = cCnt
				counter[chars] = 0
				cCnt += 1
			y.append(c2y[chars])
			counter[chars] += 1

	print("Dataset y={}, count = {}".format(c2y, counter))
	tStart = time.time()
	svc = svm.SVC(kernel='poly', gamma=0.7, C=1.0).fit(X, y)
	print("Training score:", svc.score(X,y))
	if len(sys.argv) > 1:
		for tn in sys.argv[1:]:
			X_test = denominant_color(tn, cache=None).reshape(-1)
			print(tn, "\t result:", y2c[svc.predict(X_test)[0]])