import tifffile as tiff
import numpy as np
import os
import staintools

def read_tiff(image_file, mode='rgb'):
	image = tiff.imread(image_file)
	image = image.squeeze()
	if image.shape[0] == 3:
		image = image.transpose(1, 2, 0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	return image


target_img = read_tiff(os.path.join('/home/r10user9/Documents/hhb/dataset/test_images/10078.tiff'), 'rgb')
# normalizer = staintools.ReinhardColorNormalizer()
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target_img)

train_dir = '/home/r10user9/Documents/hhb/dataset/train_images'
transform_dir = '/home/r10user9/Documents/hhb/dataset/train_vahadane_images'

for file in os.listdir(train_dir):
    print(file)
    tiff_file = os.path.join(train_dir, file)
    tiff_img = read_tiff(tiff_file, 'rgb')
    tiff_img = normalizer.transform(tiff_img)
    tiff_img = tiff_img.transpose(2, 0, 1)
    tiff.imsave(os.path.join(transform_dir, file), tiff_img)
    # break