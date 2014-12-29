# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:29:42 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""

import cv2
import numpy as np


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
#			mask = sobel/sobel.max()
			mask = (sobel/sobel.max())**2
	else:
		sobel_1d = sobel.flatten()
		mask = sobel>np.sort(sobel_1d)[TOP_RATIO*len(sobel_1d)]

		mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
							np.ones(F_SHAPE, dtype = np.uint8))

	if kwargd.get('rtn_img'):
		return (img * mask[:,:,np.newaxis]).astype(np.uint8)
	else:
		if kwargd.get('weight_mul'):
			mask *= img.shape[0] * img.shape[1] / np.sum(mask)   * 25
			mask += 0.5
			return np.repeat(img.reshape(-1,3), mask.flatten().astype(np.uint8), axis=0)
		if kwargd.get('rtn_weight'):
			return mask
		return img[np.where(mask)]

def color_sortByHue(idxs, c_hls, max_cut = True):
	idxs = sorted(idxs, key=lambda n:c_hls[n,0])
	if max_cut:
		hue = c_hls[idxs,0]
		dHue = hue[1:] - hue[:-1]
#		dHue0 = hue[0] + 1 - hue[-1] # colorsys space
		dHue0 = hue[0] + 180 - hue[-1] # cv2 space
		if dHue.max() > dHue0:
			idxs = idxs[dHue.argmax()+1:] +idxs[:dHue.argmax()+1]
	return idxs

def color_sortByLum(idxs, c_hls):
	return sorted(idxs, key=lambda n:c_hls[n,1])

def color_quantization(img, **kwargs):
	'''
	color quantization for BGR image
	'''

	K = 5
	ATTEMPTS = 10
	MAX_ITER = 10
	EPSILON = 1.0
	TOL_L = 255/K/2

	kwargd = dict(sort = 'L', K=8)
	kwargd.update(kwargs)

	K = kwargd['K']

	termcrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITER, EPSILON)
	_, _, centroids = cv2.kmeans(img.reshape(-1,3).astype(np.float32), K, None, termcrit, ATTEMPTS, cv2.KMEANS_PP_CENTERS)

	centroids = centroids.astype(np.uint8)

	c_rank = list(range(K))
	c_hls = cv2.cvtColor(centroids[:,np.newaxis], cv2.COLOR_BGR2HLS)[:,0,:]
	sort = kwargd['sort']
	if sort == 'H':
		c_rank = color_sortByHue(c_rank, c_hls, max_cut = True)
	if sort == 'L':
		c_rank = color_sortByLum(c_rank, c_hls)
#		i = j = 0
#		while j<K:
#			while (j<K) and (c_hls[c_rank[j], 1] - c_hls[c_rank[i], 1] < TOL_L): j += 1
#			if j-1>i:
#				c_rank[i:j] = color_sortByHue(c_rank[i:j], c_hls, max_cut = True)
#			i = j

	ccentroids = centroids[c_rank]

	return ccentroids


PALETTE_MIN_H = 0
PALETTE_MAX_H = 180
PALETTE_DIV_H = 16
PALETTE_STE_H = (PALETTE_MAX_H - PALETTE_MIN_H) / PALETTE_DIV_H

PALETTE_DIV_L = 8
PALETTE_STE_L = 256 / PALETTE_DIV_L
PALETTE_MIN_L = PALETTE_STE_L
PALETTE_MAX_L = 256 - 1 - PALETTE_MIN_L


def make_palette(sat = 255, display = True):

	PALETTE_H, PALETTE_L = np.mgrid[ PALETTE_MIN_H:PALETTE_MAX_H:PALETTE_STE_H,
		PALETTE_MIN_L:PALETTE_MAX_L:PALETTE_STE_L]

#	PALETTE_H += PALETTE_STE_H / 2
#	PALETTE_L += PALETTE_STE_L / 2

	PALETTE_S = np.ones(PALETTE_H.shape)*sat

	PALETTE = np.array([PALETTE_H, PALETTE_L, PALETTE_S]).T.astype(np.uint8)

	img = cv2.cvtColor(PALETTE, cv2.COLOR_HLS2BGR)

	if display:
		cv2.imshow('title', img)
		c = 0
		while not c in [27, 113]:
			c = cv2.waitKey()
		cv2.destroyAllWindows()

	return PALETTE

def hlHist(img):
	if len(img.shape) == 2:
		if img.shape[0] & 1:
			img = np.concatenate([img, np.array([[0,0,0]], dtype = np.uint8)])
		img = img.reshape(2,-1,3)
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	hist = cv2.calcHist([hls], [0,1], None,
		[PALETTE_DIV_H, PALETTE_DIV_L - 2],
		[PALETTE_MIN_H, PALETTE_MAX_H + 1,
		PALETTE_MIN_L, PALETTE_MAX_L + 1])
#	hist = hist.astype(np.uint32)
	size = np.prod(img.shape[:2])
	l = hls[:,:,1]

	hist = np.log(hist.astype(np.uint32)+1+1e-6)
	dark = np.log(np.sum(l<PALETTE_MIN_L)+1+1e-6)
	light =np.log(np.sum(l>PALETTE_MAX_L)+1+1e-6)

#	hist = hist.astype(np.uint32) / size
#	dark = np.sum(l<PALETTE_MIN_L) / size
#	light = np.sum(l>PALETTE_MAX_L) / size


	return cv2.blur(hist, (2, 2)), dark, light

def bgrOnLum(img):
	if len(img.shape) == 2:
		if img.shape[0] & 1:
			img = np.concatenate([img, np.array([[0,0,0]], dtype = np.uint8)])
		img = img.reshape(2,-1,3)
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

	vol = np.zeros((PALETTE_DIV_L - 2, 3), dtype = np.uint32)

	i = 0
	l = hls[:,:,1]
	l_low = PALETTE_MIN_L
	while l_low <  PALETTE_MAX_L:
		r = l_low<l
		r = r.__and__(l<l_low+PALETTE_STE_L)
		inrange = img[np.where(r)]
		vol[i,:] = np.average(inrange, axis = (0))
		i += 1
		l_low += PALETTE_STE_L

	return vol


import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

palette = make_palette(display = False)
palette = cv2.cvtColor(palette, cv2.COLOR_HLS2RGB)/256

def show_hlHist(img, hist, dark, light, title = None, fig = None, wait = 0.01):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if not fig:
		fig = pl.figure()
	gs = gridspec.GridSpec(4, 4)
	pl.subplot(gs[:4, :3], title = title)
	pl.imshow(img_rgb)
	pl.subplot(gs[:2, 3])
	pl.bar([0, 1], [dark, light], color=[[0.1,0.1,0.1], [0.6,0.7,0.8]])
	pl.xticks([0, 1],['Dark', 'Light'])
#	pl.imshow(hist.T, cmap = pl.cm.gray_r) # show gray map
	ax = fig.add_subplot(gs[2:, 2:], projection='3d')
	xs, zs = range(hist.shape[0]), range(hist.shape[1])
	for z in zs:
		ax.bar(xs, hist[:,z], zs = z, zdir='y', alpha=0.7,
			color = palette[z])
	if wait != 0:
		pl.show(block = wait<0)
		pl.draw()
		pl.pause(max(wait,0))

if __name__ == '__main__':
	import sys
	fig = pl.figure()
	img = cv2.imread(sys.argv[1])
	hist, dark, light = hlHist(img)
	show_hlHist(img, hist, dark, light, sys.argv[1], fig)

#	make_palette()
