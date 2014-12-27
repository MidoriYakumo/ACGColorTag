# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 13:46:34 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""




from PIL import Image, ImageDraw, ImageFont
from colorsys import rgb_to_hsv, rgb_to_hls

import numpy as np
from sklearn.feature_extraction.image import img_to_graph

import os, sys, time, glob
import h5py


def sobel(img):
	img = np.array(img.convert('RGB'))
	res = np.zeros(img.shape[:-1])

	fltr = np.array([
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1
		]).reshape(3,3)

	for i in range(1, img.shape[0]-1):
		for j in range(1, img.shape[1]-1):
			r = img[i-1:i+2,j-1:j+2,0]
			g = img[i-1:i+2,j-1:j+2,1]
			b = img[i-1:i+2,j-1:j+2,1]
			res[i, j] = (
				+ np.sum(r * fltr)**2 + np.sum(r * fltr.T)**2
				+ np.sum(g * fltr)**2 + np.sum(g * fltr.T)**2
				+ np.sum(b * fltr)**2 + np.sum(b * fltr.T)**2
				) ** 0.5

	res = np.asarray(res * 255/r.max(), dtype = np.uint8)
	return Image.fromarray(res)

def details(img, rtn_img = True):
	img = np.array(img.convert('RGB'))
	res = np.zeros(img.shape[:-1])

	fltr = np.array([
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1
		]).reshape(3,3)

	for i in range(1, img.shape[0]-1):
		for j in range(1, img.shape[1]-1):
			r = img[i-1:i+2,j-1:j+2,0]
			g = img[i-1:i+2,j-1:j+2,1]
			b = img[i-1:i+2,j-1:j+2,1]
			res[i, j] = (
				+ np.sum(r * fltr)**2 + np.sum(r * fltr.T)**2
				+ np.sum(g * fltr)**2 + np.sum(g * fltr.T)**2
				+ np.sum(b * fltr)**2 + np.sum(b * fltr.T)**2
				) ** 0.5

	LF_SIZE = 2
	lf = np.zeros(res.shape)
	for i in range(LF_SIZE, img.shape[0]-LF_SIZE):
		for j in range(LF_SIZE, img.shape[1]-LF_SIZE):
			lf[i,j] = np.sum(res[i-LF_SIZE:i+LF_SIZE+1,j-LF_SIZE:j+LF_SIZE+1])

	avg = np.average(lf)
	mask = lf>avg

	if rtn_img:
		res = img * mask[:,:,np.newaxis]
		#res = np.asarray(res * 255/res.max(), dtype = np.uint8)
		return Image.fromarray(res)
	else:
#		print(mask.shape, img.shape)
#		return np.extract(mask, img)
		return img[np.where(mask)]




#img = Image.open('/home/macrobull/workspace/python/ACGColor/test/Samail - 射命丸 文 (31787223@2458502)[東方 しゃめいまるあや 射命丸文 尻神様 ]{Photoshop}(3_5).jpg')
img = Image.open('/home/macrobull/workspace/python/ACGColor/test/small.jpg')

r = body(img, rtn_img = False)
print(r)
#r.show()
