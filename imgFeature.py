# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 13:46:34 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""

import sys

import cv2
import numpy as np



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



if __name__ == '__main__':

	fn =sys.argv[1]
	img = cv2.imread(fn)
	preview = thumbnail(img, SIZE_PREVIEW)
	thumb = thumbnail(img, SIZE_THUMB, cv2.INTER_LINEAR)

	cv2.imshow(fn+'(focusing)', focus(preview, rtn_img = True))
	body = focus(thumb)
	colors = color_quantization(body)

	draw_palette(preview, colors)
	cv2.imshow(fn+'(focused)', preview)

	colors = color_quantization(thumb)

	draw_palette(preview, colors)
	cv2.imshow(fn, preview)

	cv2.waitKey()
	cv2.destroyAllWindows()










def denominant_color(filename, **kwargs):


	cRGB = None

	cache_mode = kwargd.get('cache_mode')
	cache = kwargd.get('cache')
	if cache:
		md5=os.popen("md5sum '"+filename+"'").read().split()[0]
		if os.path.isfile(cache):
			hdf = h5py.File(cache, "r+")
			# TODO check version
			if md5 in hdf:
				cRGB = np.array(hdf[md5])
				print('[Cache hit] ' + filename)
				if not kwargd.get('rtn_img'):
					hdf.close()
					return cRGB
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
	show_img = kwargd.get('show_img')
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

		vect_sort = kwargd.get('vect_sort')
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
		kwargd['show_img'] = None

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

	if kwargd.get('rtn_img'): return preview
	if kwargd.get('show_img'): preview.show(title = filename)

	return cRGB
