import sys, os

sys.path.append('../input/hubmap-submit-06/[third_party]')
sys.path.append('../[third_party]')
#sys.path.append('/root/share1/kaggle/2022/hubmap-organ-segmentation/code/hubmap-dummy-02/[submit]/[third_party]')

from my_lib_kv import *

import tifffile as tiff
import json
import cv2
import pandas as pd
import math
import numpy as np


class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)


#--- helper ----------
def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)
	
	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)
	
	else:
		raise NotImplementedError
	
def image_show(name, image, type='bgr', resize=1):
	if type == 'rgb': image = np.ascontiguousarray(image[:,:,::-1])
	H,W = image.shape[0:2]
	
	cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  #WINDOW_NORMAL #WINDOW_GUI_EXPANDED
	cv2.imshow(name, image) #.astype(np.uint8))
	cv2.resizeWindow(name, round(resize*W), round(resize*H))


##--------------------------------------------------------------------------------------
organ_meta = dotdict(
	kidney = dotdict(
		label = 1,
		um    = 0.5000,
		ftu   ='glomeruli',
	),
	prostate = dotdict(
		label = 2,
		um    = 6.2630,
		ftu   ='glandular acinus',
	),
	largeintestine = dotdict(
		label = 3,
		um    = 0.2290,
		ftu   ='crypt',
	),
	spleen = dotdict(
		label = 4,
		um    = 0.4945,
		ftu   ='white pulp',
	),
	lung = dotdict(
		label = 5,
		um    = 0.7562,
		ftu   ='alveolus',
	),
)



organ_to_label = {k: organ_meta[k].label for k in organ_meta.keys()}
label_to_organ = {v:k for k,v in organ_to_label.items()}
num_organ=5
#['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']


def read_tiff(image_file, mode='rgb'):
	image = tiff.imread(image_file)
	image = image.squeeze()
	if image.shape[0] == 3:
		image = image.transpose(1, 2, 0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	return image

def read_json_as_list(json_file):
	with open(json_file) as f:
		j = json.load(f)
	return j


# # --- rle ---------------------------------
def rle_decode(rle, height, width , fill=255, dtype=np.uint8):
	s = rle.split()
	start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	start -= 1
	mask = np.zeros(height*width, dtype=dtype)
	for i, l in zip(start, length):
		mask[i:i+l] = fill
	mask = mask.reshape(width,height).T
	mask = np.ascontiguousarray(mask)
	return mask


def rle_encode(mask):
	m = mask.T.flatten()
	m = np.concatenate([[0], m, [0]])
	run = np.where(m[1:] != m[:-1])[0] + 1
	run[1::2] -= run[::2]
	rle =  ' '.join(str(r) for r in run)
	return rle


#
# # --draw ------------------------------------------
def mask_to_inner_contour(mask):
	mask = mask>0.5
	pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
	contour = mask & (
			(pad[1:-1,1:-1] != pad[:-2,1:-1]) \
			| (pad[1:-1,1:-1] != pad[2:,1:-1]) \
			| (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
			| (pad[1:-1,1:-1] != pad[1:-1,2:])
	)
	return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
	contour =  mask_to_inner_contour(mask)
	if thickness==1:
		image[contour] = color
	else:
		r = max(1,thickness//2)
		for y,x in np.stack(np.where(contour)).T:
			cv2.circle(image, (x,y), r, color, lineType=cv2.LINE_4 )
	return image

def result_to_overlay(image, mask, probability=None, **kwargs):
 
	
	H,W,C= image.shape
	if mask is None:
		mask = np.zeros((H,W),np.float32)
	if probability is None:
		probability = np.zeros((H,W),np.float32)
		
	o1 = np.zeros((H,W,3),np.float32)
	o1[:,:,2] = mask
	o1[:,:,1] = probability
	
	o2 = image.copy()
	o2 = o2*0.5
	o2[:,:,1] += 0.5*probability
	o2 = draw_contour_overlay(o2, mask, color=(0,0,1), thickness=max(3,int(7*H/1500)))
	
	#---
	o2,image,o1 = [(m*255).astype(np.uint8) for m in [o2,image,o1]]
	if kwargs.get('dice_score',-1)>=0:
		draw_shadow_text(o2,'dice=%0.5f'%kwargs.get('dice_score'),(20,80),2.5,(255,255,255),5)
	if kwargs.get('d',None) is not None:
		d = kwargs.get('d')
		draw_shadow_text(o2,d['id'],(20,140),1.5,(255,255,255),3)
		draw_shadow_text(o2,d.organ+'(%s)'%(organ_meta[d.organ].ftu),(20,190),1.5,(255,255,255),3)
		draw_shadow_text(o2,'%0.1f um'%(d.pixel_size),(20,240),1.5,(255,255,255),3)
		s100 = int(100/d.pixel_size)
		sx,sy = W-s100-40,H-80
		cv2.rectangle(o2,(sx,sy),(sx+s100,sy+s100//2),(0,0,0),-1)
		draw_shadow_text(o2,'100um',(sx+8,sy+40),1,(255,255,255),2)
		pass
	
	#draw_shadow_text(image,'input',(5,15),0.6,(1,1,1),1)
	#draw_shadow_text(im_paste,'predict',(5,15),0.6,(1,1,1),1)

	overlay = np.hstack([o2,image,o1])
	return overlay

# def result_to_overlay1(probability, truth):
# 	H,W = probability.shape
# 	thickness = max(int((H+W)/2/512*5),2)
#
# 	o1 = np.zeros((H,W,3),np.float32)
# 	o1[:,:,2] = truth
# 	o1[:,:,1] = probability
#
# 	o2 = np.zeros((H,W,3),np.float32)
# 	o2 = o2*0.5
# 	o2[:,:,1] += 0.5*probability
# 	o2 = draw_contour_overlay(o2, truth, color=(0,0,1), thickness=thickness)
#
# 	overlay = np.hstack([o2,o1])
# 	return overlay


# --lb metric ------------------------------------------
# https://www.kaggle.com/competitions/hubmap-organ-segmentation/overview/supervised-ml-evaluation

def compute_dice_score(probability, mask):
	N = len(probability)
	p = probability.reshape(N,-1)
	t = mask.reshape(N,-1)
	
	p = p>0.5
	t = t>0.5
	uion = p.sum(-1) + t.sum(-1)
	overlap = (p*t).sum(-1)
	dice = 2*overlap/(uion+0.0001)
	return dice
