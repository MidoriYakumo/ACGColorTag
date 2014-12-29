# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:29:42 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""

import cv2
import numpy as np

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

def hlVector(img):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	hist = cv2.calcHist([hls], [0,1], None,
		[PALETTE_DIV_H, PALETTE_DIV_L - 2],
		[PALETTE_MIN_H, PALETTE_MAX_H + 1,
		PALETTE_MIN_L, PALETTE_MAX_L + 1])
#	hist = hist.astype(np.uint32)
	size = np.prod(img.shape[:2])
	hist = hist.astype(np.uint32) / size
	l = hls[:,:,1]
	dark = np.sum(l<PALETTE_MIN_L) / size
	light = np.sum(l>PALETTE_MAX_L) / size
	return hist, dark, light


import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

palette = make_palette(display = False)
palette = cv2.cvtColor(palette, cv2.COLOR_HLS2RGB)/256

def show_hlFeature(img, hist, dark, light, title = None, fig = None, wait = 0.01):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if not fig:
		fig = pl.figure()
	gs = gridspec.GridSpec(4, 4)
	pl.subplot(gs[:4, :3], title = title)
	pl.imshow(img_rgb)
	pl.subplot(gs[:2, 3])
	pl.bar([0, 1], [dark, light], color=[[0.1,0.1,0.1], [0.6,0.7,0.8]])
	pl.xticks([0, 1],['Dark', 'Light'])
	ax = fig.add_subplot(gs[2:, 2:], projection='3d')
	xs, zs = range(hist.shape[0]), range(hist.shape[1])
	for z in zs:
		ax.bar(xs, hist[:,z], zs = z, zdir='y', alpha=0.7,
			color = palette[z])
	if wait != 0:
		pl.show(block = wait<0)
		pl.draw()
		pl.pause(wait)

if __name__ == '__main__':
	import sys
	fig = pl.figure()
	img = cv2.imread(sys.argv[1])
	hist, dark, light = hlVector(img)
	show_hlFeature(img, hist, dark, light, sys.argv[1], fig)
	img = cv2.imread(sys.argv[2])
	hist, dark, light = hlVector(img)
	show_hlFeature(img, hist, dark, light, sys.argv[2], fig)
