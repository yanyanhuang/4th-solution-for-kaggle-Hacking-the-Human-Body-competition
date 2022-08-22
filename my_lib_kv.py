import numpy as np
import cv2

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