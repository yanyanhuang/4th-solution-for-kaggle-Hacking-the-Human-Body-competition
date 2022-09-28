import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
import pandas as pd
import os
import numpy as np
from stainnet import StainNet
import torch
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

def enc2mask(mask_rle, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image=image[np.newaxis, ...]
    image=torch.from_numpy(image)
    return image

def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1,2,0))
    return image

train_df = pd.read_csv('/home/r10user9/Documents/hhb/dataset/train.csv')

image_dir = '/home/r10user9/Documents/hhb/dataset/train_images'

list = train_df[train_df['organ'] == 'kidney'].id.values
# list = [10044, 10912, 10971, 13942, 16609, 20247, 23640, 29213, 4944, 6390, 7397, 7902]
# for i in range(len(lung_list)):
for i in range(10):
    image_id = list[i]

    source = os.path.join(image_dir, str(image_id) + '.tiff')
    target = os.path.join('/home/r10user9/Documents/hhb/dataset/test_images/10078.tiff')
    source = read_tiff(source, 'rgb')
    # target = read_tiff(target, 'rgb')
    # image = source.astype(np.float32)/255

    # normalizer = staintools.ReinhardColorNormalizer()
    # normalizer = staintools.StainNormalizer(method='vahadane')
    # normalizer.fit(target)
    # transform_image = normalizer.transform(source)


    # stainnet = StainNet().cuda()
    # stainnet.load_state_dict(torch.load('/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/StainNet-3x0_best_psnr_layer3_ch32.pth'))

    # transform_image = un_norm(stainnet(norm(source).cuda()))
    # fft_image = np.abs(np.fft.fft2(image))

    d = train_df[train_df['id'] == image_id]
    rle = d.rle.item()
    mask = enc2mask(rle,(source.shape[1],source.shape[0])) if rle is not None else None
    print(image_id)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1); plt.imshow(source); plt.axis('OFF'); plt.title('image')
    # plt.subplot(1, 3, 2); plt.imshow(target); plt.axis('OFF'); plt.title('mask')
    plt.subplot(1, 3, 2); plt.imshow(source); plt.imshow(mask*255, alpha=0.4); plt.axis('OFF'); plt.title('overlay')
    # plt.subplot(1, 3, 3); plt.imshow(transform_image); plt.axis('OFF'); plt.title('mask')

    plt.show()