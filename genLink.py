# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 10:21:46 2014
Project	:Python-Project
Version	:0.0.1
@author	:macrobull (http://github.com/macrobull)

"""

import os, sys

f=open('Chars', 'r')
ESCAPE = f.readline().split()[1:]
#print(ESCAPE)

cCnt = {}
mapper = {}
for l in f:
	info = l.split()
	for alias in info[1:]:
		mapper[alias] = info[0]
	cCnt[info[0]] = 0

f.close()

#print(mapper)

os.popen('rm "' + sys.argv[2] + '/"*')

for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
	for ff in filenames:
		fn, ext = os.path.splitext(ff)
		for es in ESCAPE:
			fn = fn.replace(es, ' ')

		match = 0
		char = None
		for tag in fn.split():
			if tag in mapper:
				c = mapper[tag]
				if char != c:
					match += 1
					char = c

		if match == 1:
			os.popen("ln -s '"+dirpath + os.sep + ff+"' '"
				+ sys.argv[2] + os.sep + char + '.' +str(cCnt[char]) +
				ext + "'")
			cCnt[char] += 1

#sys.argv[1]
#sys.argv[2]

