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
import os, sys, time
import h5py


from sklearn import svm

__version__ = '0.0.1'

cSize = 30
cMargin = 10
FONT = os.environ['HOME'] + "/.fonts/Android/DroidSansFallback.ttf"


K = 8
#KC = 3
#LOWSAT = 50

def denominant_color(filename, showImg = True, cache = 'cache.hdf5', **kwargs):
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


	def putCircle(xy, color, **kwargs):
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
	thumb = img.resize((int(width/downscale), int(height/downscale)), Image.NEAREST).convert('RGB')
	downscale = min(img.size)/512
	preview = img.resize((int(width/downscale), int(height/downscale)), Image.ANTIALIAS)

	clt = KMeans(K, n_jobs = -1,
#		max_iter = 200, n_init = 8,
		init = "k-means++")
	clt.fit(np.array(thumb).reshape((-1,3)))


	cScale = np.array(2)
	canvas = Image.new('RGBA', preview.size * cScale)
	painter = ImageDraw.Draw(canvas)

	rRank = list(range(K))
	cRGB = clt.cluster_centers_
	cHLS = np.array([rgb_to_hls(*c) for c in cRGB])

	'''
	rLight = cHLS[:,1].argmax()
	rDark  = cHLS[:,1].argmin()
#	cLight = cRGB[rLight]
#	cDark = cRGB[rDark]

	cCnt = np.zeros(K)
	for c in clt.labels_: cCnt[c] += 1
	cMax = cCnt.max()


	for i in range(K - 1, -1, -1):
		if (cHLS[rLight, 1] - cHLS[i,1]< LOWSAT) or (cHLS[i,1] - cHLS[rDark, 1]< LOWSAT):
			rRank.pop(i)

	while len(rRank)<KC: rRank.append(rLight)
	rRank = rRank[:KC]
	print(rLight, rDark, rRank)

	rRank = [rLight, rDark] + rRank
	rRank = sorted(rRank, key=lambda n:cCnt[n], reverse=True)

	#Original color
	for i, color in enumerate(cRGB):
		putCircle(((2*i+1)*(cMargin + cSize), (cSize*3 + cMargin*3)), color, int(cCnt[i]*255/cMax))
		putCircle(((2*i+1)*(cMargin + cSize), (cSize*5 + cMargin*5)), color)
	'''

#	#Sort with Hue
#	rRank = sorted(rRank, key=lambda n:cHLS[n,0])
#
#	#Make it best cut on Hue
#	cDL = cHLS[rRank][1:,0] - cHLS[rRank][:-1,0]
##	print(1 + cHLS[rRank][0,0] - cHLS[rRank][-1,0], cDL)
#	if cDL.max() > 1 + cHLS[rRank][0,0] - cHLS[rRank][-1,0]:
#		rRank = rRank[cDL.argmax()+1:] + rRank[:cDL.argmax()+1]

	#Sort with Luminance
	rRank = sorted(rRank, key=lambda n:cHLS[n,1])

	#Pass to output
	cRGB = cRGB[rRank]
	for i, color in enumerate(cRGB):
		putCircle(((2*i+1)*(cMargin + cSize), (cSize + cMargin)), color)

	preview = Image.alpha_composite(preview, canvas.resize(preview.size, Image.ANTIALIAS))

	if cache:
		hdf.create_dataset(md5, data=cRGB)
		hdf.close()

	if 'rtn_img' in kwargs and kwargs['rtn_img']: return preview
	if showImg: preview.show()
	return cRGB

if __name__ == '__main__':
	X = []
	y = []
	c2y = {}
	y2c = {}
	cCnt = 0
	counter = {}
	imgdir = 'img/'
	os.popen('killall display')
	for fn in os.listdir(imgdir):
#		try:
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
#		except Exception as e:
#			print(fn, e)

	print("Dataset y={}, count = {}".format(c2y, counter))
	tStart = time.time()
	svc = svm.SVC(kernel='poly', gamma=0.7, C=1.0).fit(X, y)
	print("Training score:", svc.score(X,y))
	if len(sys.argv) > 1:
		for i, tn in enumerate(sys.argv[1:]):
			X_test = denominant_color(tn, showImg = False, cache=None)
			y_test = y2c[svc.predict(X_test.reshape(-1))[0]]
			print("{} in {}.".format(y_test, tn))
			img = denominant_color(tn, cache=None, rtn_img = True)
			canvas = Image.new('RGBA', img.size)
			painter = ImageDraw.Draw(canvas)
			font = ImageFont.truetype(FONT, 72)
			painter.text((cMargin +2, cSize * 2 + cMargin * 3 +2),
				y_test, font=font,
				fill = (99,99,99,180))
			painter.text((cMargin, cSize * 2 + cMargin * 3),
				y_test, font=font,
				fill = tuple([int(c) for c in X_test[int(len(X_test)*2/3)]]))
			img = Image.alpha_composite(img, canvas)
			img.save("/tmp/{}.{}.jpg".format(i, y_test), "JPEG")
			os.popen("feh '/tmp/{}.{}.jpg'".format(i, y_test))
