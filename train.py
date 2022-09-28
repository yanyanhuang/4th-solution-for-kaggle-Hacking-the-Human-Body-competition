from base64 import decode
import os
import cv2
import time
import random
import math

import torch
from torch import nn
import torch.cuda.amp as amp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from torch.utils.data import SequentialSampler
import torch.nn.functional as F
# from torchmetrics.functional import dice_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from swin import Net
from kaggle_hubmap_kv3 import *
import importlib
import matplotlib.pyplot as plt

is_amp = True
import logging
import pandas as pd
from sklearn.model_selection import KFold

import numpy as np
import staintools
# from itertools import repeat
# import collections.abc


import warnings
warnings.filterwarnings('ignore')

os.makedirs('/home/r10user9/Documents/hhb/coatnet_baseline/result', exist_ok=True)
os.makedirs('/home/r10user9/Documents/hhb/coatnet_baseline/checkpoint', exist_ok=True)

root_dir = '/home/r10user9/Documents/hhb/coatnet_baseline/'
pretrain_dir = '/kaggle/input/swin-tiny-small-22k-pretrained/'

TRAIN = '/home/r10user9/Documents/hhb/dataset/train_images/'
# MASKS = '../input/hubmap-2022-256x256/masks/'
LABELS = '/home/r10user9/Documents/hhb/dataset/train.csv'

#%%
organ_threshold = {
    'Hubmap': {
        0        : 0.40,
        1        : 0.40,
        2        : 0.40,
        3        : 0.40,
        4        : 0.10,
    },
    'HPA': {
        0        : 0.50,
        1        : 0.50,
        2        : 0.50,
        3        : 0.50,
        4        : 0.10,
    },
}

def do_tta_batch(image, mask):
    
    batch = { #<todo> multiscale????
        'image': torch.stack([
            image,
            torch.flip(image,dims=[1]),
            torch.flip(image,dims=[2]),
        ]),
        'mask': torch.stack([
            mask,
            torch.flip(mask,dims=[1]),
            torch.flip(mask,dims=[2]),
        ])
        # 'organ': torch.Tensor(
        #     [[organ_to_label[organ]]]*3
        # ).long()
    }
    return batch

def undo_tta_batch(probability):
    probability[0] = probability[0]
    probability[1] = torch.flip(probability[1],dims=[1])
    probability[2] = torch.flip(probability[2],dims=[2])
    probability = probability.mean(0, keepdims=True)
    probability = probability[0,0].float()
    return probability

# 'kidney' : 0,
# 'prostate' : 1,
# 'largeintestine' : 2,
# 'spleen' : 3,
# 'lung' : 4

def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit,mask)
    return loss


def load_net_test(model):
    print('\tload %s ... '%(model.module),end='',flush=True)
    M = importlib.import_module(model.module)
    num = len(model.checkpoint)
    net = []
    for f in range(num):
        n = M.Net(**model.param)
        n.load_state_dict(
            torch.load(model.checkpoint[f], map_location=lambda storage, loc: storage) ['state_dict'],
            strict=False)
        n.cuda()
        n.eval()
        net.append(n)
        
    print('ok!')
    return net


def enc2mask(mask_rle, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def image_to_tensor(image, mode='bgr'): #image mode
    if mode=='bgr':
        image = image[:,:,::-1]
    x = image
    x = x.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


def mask_to_tensor(mask):
    x = mask
    x = torch.tensor(x, dtype=torch.float)
    return x


tensor_list = ['mask', 'image', 'organ']

def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_list:
            v = torch.stack(v)
        d[k] = v

    d['mask'] = d['mask'].unsqueeze(1)
    d['organ'] = d['organ'].reshape(-1)
    return d
    
    
# def message(mode='print'):
#     asterisk = ' '
#     if mode==('print'):
#         loss = batch_loss
#     if mode==('log'):
#         loss = train_loss
#         if (iteration % iter_save == 0): asterisk = '*'

#     text = \
#         ('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
#         '%4.3f  %4.3f  %4.4f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f   | '%(*valid_loss,) + \
#         '%4.3f  %4.3f   | '%(*loss,) + \
#         '%s' % ((time.time() - start_timer))

#     return text

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

#%% Random Choice
def valid_augment5(image, mask, organ):
    #image, mask  = do_crop(image, mask, image_size, xy=(None,None))
    return image, mask

def train_augment5b(image, mask, organ):
    image, mask = do_random_flip(image, mask)
    image, mask = do_random_rot90(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_noise(image, mask, mag=0.1),
        lambda image, mask: do_random_contast(image, mask, mag=0.40),
        lambda image, mask: do_random_hsv(image, mask, mag=[0.40, 0.40, 0])
    ], 2): image, mask = fn(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_rotate_scale(image, mask, angle=45, scale=[0.50, 2.0]),
    ], 1): image, mask = fn(image, mask)

    return image, mask

def stain_aug(image):

    for fn in np.random.choice([
        lambda image: (image),
        lambda image: normalizer.transform(image),
    ], 1): image = fn(image)
    return image

target_img = read_tiff(os.path.join('/home/r10user9/Documents/hhb/dataset/test_images/10078.tiff'), 'rgb')
normalizer = staintools.ReinhardColorNormalizer()
normalizer.fit(target_img)

def do_random_flip(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    if np.random.rand()>0.5:
        image = image.transpose(1,0,2)
        mask = mask.transpose(1,0)
    
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask

def do_random_rot90(image, mask):
    r = np.random.choice([
        0,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ])
    if r==0:
        return image, mask
    else:
        image = cv2.rotate(image, r)
        mask = cv2.rotate(mask, r)
        return image, mask
    
def do_random_contast(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag
    image = image * alpha
    image = np.clip(image,0,1)
    return image, mask

def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
    image = (image*255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
    s =  s*(1 + random.uniform(-1,1)*mag[1])
    v =  v*(1 + random.uniform(-1,1)*mag[2])

    hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32)/255
    return image, mask

def do_random_noise(image, mask, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1,1, (height, width,1))*mag
    image = image + noise
    image = np.clip(image,0,1)
    return image, mask

def do_random_rotate_scale(image, mask, angle=30, scale=[0.8,1.2] ):
    angle = np.random.uniform(-angle, angle)
    scale = np.random.uniform(*scale) if scale is not None else 1
    
    height, width = image.shape[:2]
    center = (height // 2, width // 2)
    
    transform = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask  = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask

#%% Dataset
# image_size = 1536
# image_size = 1024

class HubmapDataset(Dataset):
    def __init__(self, df, augment=None, train=None, group_id=0, image_size=None):
        # if group_id == 0:
        #     df = df[df['organ'] != 'lung'].reset_index(drop=True)
        # elif group_id == 1:
        #     df = df[df['organ'] == 'lung'].reset_index(drop=True)
        self.df = df
        self.image_size = image_size
        self.augment = augment
        self.length = len(self.df)
        ids = pd.read_csv(LABELS).id.astype(str).values
        # self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('.')[0] in ids]
        self.organ_to_label = {'kidney' : 0,
                               'prostate' : 1,
                               'largeintestine' : 2,
                               'spleen' : 3,
                               'lung' : 4}
        self.train = train
        self.train_dirs = [TRAIN, TRAIN.replace('train_images', 'train_reinhard_images')]
        # import ipdb;ipdb.set_trace()

    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # import ipdb;ipdb.set_trace()
        # fname = self.fnames[index]
        d = self.df.iloc[index]
        data_source = d.data_source
        fname = d.id
        organ = self.organ_to_label[d.organ]

        if self.train == True:
            train_dir = self.train_dirs[np.random.randint(2)]
        else:
            train_dir = self.train_dirs[0]

        tiff_file = os.path.join(train_dir, str(fname) + '.tiff')
        tiff = read_tiff(tiff_file, 'rgb')
        # stain_aug
        # if self.train == True:
        #     tiff = stain_aug(tiff)
        image = tiff.astype(np.float32)/255

        rle = d.rle
        mask = enc2mask(rle,(tiff.shape[1],tiff.shape[0])) if rle is not None else None

        # image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        
        # image = image.astype(np.float32)/255
        # mask  = mask.astype(np.float32)/255

        s = d.pixel_size/0.4 * (self.image_size/3000)
        image = cv2.resize(image,dsize=(self.image_size,self.image_size),interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(self.image_size,self.image_size),interpolation=cv2.INTER_LINEAR)

        if self.augment is not None:
            image, mask = self.augment(image, mask, organ)


        r ={}
        r['index']= index
        r['id'] = fname
        r['organ'] = torch.tensor([organ], dtype=torch.long)
        r['image'] = image_to_tensor(image, 'rgb')
        r['mask' ] = mask_to_tensor(mask>0.5)
        r['data_source'] = data_source
        return r

#%% Configuration
# cfg = dict(

#         #configs/_base_/models/upernet_swin.py
#         basic = dict(
#             swin=dict(
#                 embed_dim=96,
#                 depths=[2, 2, 6, 2],
#                 num_heads=[3, 6, 12, 24],
#                 window_size=7,
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 qk_scale=None,
#                 drop_rate=0.,
#                 attn_drop_rate=0.,
#                 drop_path_rate=0.3,
#                 ape=False,
#                 patch_norm=True,
#                 out_indices=(0, 1, 2, 3),
#                 use_checkpoint=False
#             ),

#         ),

#         #configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
#         swin_tiny_patch4_window7_224=dict(
#             checkpoint = pretrain_dir+'/swin_tiny_patch4_window7_224_22k.pth',

#             swin = dict(
#                 embed_dim=96,
#                 depths=[2, 2, 6, 2],
#                 num_heads=[3, 6, 12, 24],
#                 window_size=7,
#                 ape=False,
#                 drop_path_rate=0.3,
#                 patch_norm=True,
#                 use_checkpoint=False,
#             ),
#             upernet=dict(
#                 in_channels=[96, 192, 384, 768],
#             ),
#         ),

#         #/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py
#         swin_small_patch4_window7_224_22k=dict(
#             checkpoint = pretrain_dir+'/swin_small_patch4_window7_224_22k.pth',

#             swin = dict(
#                 embed_dim=96,
#                 depths=[2, 2, 18, 2],
#                 num_heads=[3, 6, 12, 24],
#                 window_size=7,
#                 ape=False,
#                 drop_path_rate=0.3,
#                 patch_norm=True,
#                 use_checkpoint=False
#             ),
#             upernet=dict(
#                 in_channels=[96, 192, 384, 768],
#             ),
#         ),
#     )

#%% Folds
def make_fold(fold=0):
    df = pd.read_csv(LABELS)

    # num_fold = 5
    # skf = KFold(n_splits=num_fold, shuffle=True,random_state=42)

    # df.loc[:,'fold']=-1
    # for f,(t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
    #     df.iloc[v_idx,-1]=f

    # #check
    # if 0:
    #     for f in range(num_fold):
    #         train_df=df[df.fold!=f].reset_index(drop=True)
    #         valid_df=df[df.fold==f].reset_index(drop=True)

    #         print('fold %d'%f)
    #         t = train_df.organ.value_counts().to_dict()
    #         v = valid_df.organ.value_counts().to_dict()
    #         for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
    #             print('%32s %3d (%0.3f)  %3d (%0.3f)'%(k,t.get(k,0),t.get(k,0)/len(train_df),v.get(k,0),v.get(k,0)/len(valid_df)))

    #         print('')
    #         zz=0

    # train_df=df[df.fold!=fold].reset_index(drop=True)
    # valid_df=df[df.fold==fold].reset_index(drop=True)

    train_txt = pd.read_csv(f'splits_5/fold_{fold}.txt', header=None)
    valid_txt = pd.read_csv(f'splits_5/valid_{fold}.txt', header=None)
    train_df = df[df['id'].isin(train_txt.values.squeeze())].reset_index(drop=True)
    valid_df = df[df['id'].isin(valid_txt.values.squeeze())].reset_index(drop=True)
    # import ipdb;ipdb.set_trace()
    return train_df,valid_df

#%% Validation

def validate(n, valid_loader, epoch):

    valid_num = 0
    valid_probability = []
    valid_mask = []
    valid_loss = 0
    organ_list = []
    global best_dice
    global best_valepoch

    n = n.eval()
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):
        organ = batch['organ'][0].item()
        organ_list.append(organ)
        data_source = batch['data_source'][0]
        # import ipdb;ipdb.set_trace()
        val_batch = do_tta_batch(batch['image'][0], batch['mask'][0])
        # val_batch['image'] = val_batch['image'].cuda()
        # import ipdb;ipdb.set_trace()
        batch_size = len(batch['index'])
        valid_mask.append(batch['mask'].data.cpu().numpy())
        # batch['image'] = batch['image'].cuda()
        batch['image'] = val_batch['image'].cuda()
        batch['mask' ] = val_batch['mask' ].cuda()
        batch['organ'] = batch['organ'].cuda()
        # import ipdb;ipdb.set_trace()


        # net.output_type = ['loss', 'inference']
        use = 0
        probability = 0
        with torch.no_grad():
            with amp.autocast(enabled = is_amp):
                # for net in all_net:
                #     for n in net:
                use += 1
                n = n.eval()
                n.output_type = ['inference', 'loss']
                output = n(batch)
                probability += output['probability']
                loss0  = output['bce_loss'].mean()
                probability = undo_tta_batch(probability/use)
        # import ipdb;ipdb.set_trace()
        valid_probability.append(probability.data.cpu().numpy()[np.newaxis, :] > organ_threshold[data_source][organ])
        # valid_mask.append(batch['mask'].data.cpu().numpy())
        valid_num += batch_size
        valid_loss += batch_size*loss0.item()

        #debug
        # if 0 :
        #     pass
        #     organ = batch['organ'].data.cpu().numpy()
        #     image = batch['image']
        #     mask  = batch['mask']
        #     probability  = output['probability']

        #     for b in range(batch_size):
        #         m = tensor_to_image(image[b])
        #         t = tensor_to_mask(mask[b,0])
        #         p = tensor_to_mask(probability[b,0])
        #         overlay = result_to_overlay(m, t, p )

        #         text = label_to_organ[organ[b]]
        #         draw_shadow_text(overlay,text,(5,15),0.7,(1,1,1),1)

        #         image_show_norm('overlay',overlay,min=0,max=1,resize=1)
        #         cv2.waitKey(0)

        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),(time.time() - start_timer)),end='',flush=True)

    assert(valid_num == len(valid_loader.dataset))

    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)

    loss = valid_loss/valid_num
    # import ipdb;ipdb.set_trace()
    dice = compute_dice_score(probability, mask)
    mdice = dice.mean()
    if mdice > best_dice:
        best_dice = mdice
        best_valepoch = epoch
    # import ipdb;ipdb.set_trace()
    organ_dice = []
    for i in range(5):
        organ_dice.append(np.mean(dice[np.argwhere(np.array(organ_list) == i)]))
    
    return [mdice, loss,  best_dice, best_valepoch] + organ_dice

#%% Initialization
def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

# def load_net(model):
#     print('\tload %s ... '%(model.module),end='',flush=True)
#     M = importlib.import_module(model.module)
#     # num = len(model.checkpoint)
#     net = []
#     # for f in range(num):
#     n = M.Net(**model.param)
#     # import ipdb;ipdb.set_trace()
#     if len(model.checkpoint) > 0:
#         n.load_state_dict(torch.load(model.checkpoint[0], map_location=lambda storage, loc: storage),strict=False)
#         # n.load_state_dict({k.replace('encoder_decoders.0.0', 'encoders_mpvit.0'):v for k, v in torch.load(model.checkpoint[0], map_location=lambda storage, loc: storage).items()}, strict=False)
#         # n.load_state_dict({k.replace('encoder_decoders.1.0', 'encoders_mpvit.1'):v for k, v in torch.load(model.checkpoint[0], map_location=lambda storage, loc: storage).items()}, strict=False)
#         # n.load_state_dict({k.replace('encoder_decoders.0.1', 'decoders_daformer.0'):v for k, v in torch.load(model.checkpoint[0], map_location=lambda storage, loc: storage).items()}, strict=False)
#         # n.load_state_dict({k.replace('encoder_decoders.1.1', 'decoders_daformer.1'):v for k, v in torch.load(model.checkpoint[0], map_location=lambda storage, loc: storage).items()}, strict=False)
#     # for para in n.encoders_mpvit.parameters():
#     #     para.requires_grad = False
#     n.cuda()
#     n.train()
#     net.append(n)
        
#     print('ok!')
#     return net
def load_net(model):
    print('\tload %s ... '%(model.module),end='',flush=True)
    M = importlib.import_module(model.module)
    # num = len(model.checkpoint)
    net = []
    # for f in range(num):
    n = M.Net(**model.param)
    # import ipdb;ipdb.set_trace()
    if len(model.checkpoint) > 0:
        n.load_state_dict(torch.load(model.checkpoint[0], map_location=lambda storage, loc: storage),strict=False)
    if len(model.checkpoint1) > 0:
        n.load_state_dict(torch.load(model.checkpoint1[0], map_location=lambda storage, loc: storage),strict=False)
    # for para in n.encoders_coat.parameters():
    #     para.requires_grad = False
    n.cuda()
    n.train()
    net.append(n)
        
    print('ok!')
    return net

from coat import *
from daformer import *

best_dice = 0
best_valepoch = 0

def run(fold, patience, start_lr, gpu, group_id, image_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
    # fold = 0
    # image_size = image_size
    out_dir = os.path.join(root_dir, f'result/mpvit_daformer_unet_{image_size}_2model_2stainnorm_lova/fold-{fold}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # import ipdb;ipdb.set_trace()
    # print(root_dir)
    initial_checkpoint = None

    # start_lr   = 4e-5 #0.0001
    batch_size = 2 #32 #32


    ## setup  ----------------------------------------
    # for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)

        
    log = open(out_dir+f'/log.train.fold-{fold}.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % ('Swin', '-' * 64))
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_df, valid_df = make_fold(fold)
    # import ipdb;ipdb.set_trace()

    train_dataset = HubmapDataset(train_df, train_augment5b, train=True, group_id=group_id, image_size=image_size)
    valid_dataset = HubmapDataset(valid_df, valid_augment5, train=False, group_id=group_id, image_size=image_size)

    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size  = 1,
        drop_last   = True,
        num_workers = 2,
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn = null_collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 1,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = False,
        collate_fn = null_collate,
    )


    log.write('fold = %s\n'%str(fold))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')

    scaler = amp.GradScaler(enabled = is_amp)


    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_pvtv2_daformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/pvt_v2_b4.pth'},
    #         checkpoint = [
    #             '../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_cswin_upernet',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/upernet_cswin_small.pth'},
    #         checkpoint = [
    #             '../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_coatsmall_daformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/coat_small_7479cf9b.pth'},
    #         checkpoint = [
    #             '../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_daformer_coat',
    #         param={'encoder': coat_lite_medium, 'decoder': daformer_conv3x3, 'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/coat_lite_medium_384x384_f9129688.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt'
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_coat_unet',
    #         param={'encoder': coat_lite_medium, 'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/coat_lite_medium_384x384_f9129688.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_1024_2model_stainaug/fold-0/fold-0_207.00000000018042.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_1024_2model_stainaug/fold-1/fold-1_149.99999999998744.pt'
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_1024_2model_stainaug/fold-2/fold-2_181.9999999998965.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_1024_2model_stainaug/fold-3/fold-3_117.0000000000373.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_1024_2model_stainaug/fold-4/fold-4_102.00000000002004.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_coat_daformer_unet',
    #         param={'encoder': coat_lite_medium, 'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/coat_lite_medium_384x384_f9129688.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt'
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #         checkpoint1 = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_unet_1024_2model_2stainnorm/fold-0/fold-0_123.00000000006584.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_unet_1024_2model_2stainnorm/fold-1/fold-1_146.99999999999596.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_unet_1024_2model_2stainnorm/fold-2/fold-2_79.99999999999471.pt'
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_unet_1024_2model_2stainnorm/fold-3/fold-3_40.999999999995744.pt'
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/coat_unet_1024_2model_2stainnorm/fold-4/fold-4_66.99999999997975.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_mit_daformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/segformer.b3.512x512.ade.160k.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_stainaug/fold-0/fold-0_253.00000000024318.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_stainaug/fold-1/fold-1_193.99999999986238.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_stainaug/fold-2/fold-2_271.9999999996407.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_stainaug/fold-3/fold-3_216.999999999797.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_stainaug/fold-4/fold-4_278.9999999996208.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_mit_unet',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/segformer.b4.512x512.ade.160k.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt',
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_mit_daformer_unet',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/segformer.b3.512x512.ade.160k.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt',
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #         checkpoint1 = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_unet_1024_2model_2stainnorm/fold-0/fold-0_40.000000000000355.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_unet_1024_2model_2stainnorm/fold-1/fold-1_42.99999999999405.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_unet_1024_2model_2stainnorm/fold-2/fold-2_93.00000000000968.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_unet_1024_2model_2stainnorm/fold-3/fold-3_226.9999999997686.pt',
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/mit_unet_1024_2model_2stainnorm/fold-4/fold-4_145.00000000000165.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_pvtv2_unet',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/pvt_v2_b4.pth'},
    #         checkpoint = [
    #             '../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_mpvit_daformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/upernet_mpvit_small.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_mpvit_unet',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/upernet_mpvit_small.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt',
    #             '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #     ),
    # ]
    model = [
        dotdict(
            is_use = 1,
            module = 'model_mpvit_daformer_unet',
            param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/upernet_mpvit_small.pth'},
            checkpoint = [
                '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
            ],
            checkpoint1 = [
                '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_unet_1024_2model_2stainnorm/fold-0/fold-0_241.0000000002268.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_unet_1024_2model_2stainnorm/fold-1/fold-1_88.00000000000392.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_unet_1024_2model_2stainnorm/fold-2/fold-2_151.99999999998175.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_unet_1024_2model_2stainnorm/fold-3/fold-3_273.999999999635.pt',
                # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_unet_1024_2model_2stainnorm/fold-4/fold-4_68.99999999998205.pt'
            ],
        ),
    ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_mscan_daformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/segnext_large_512x512_ade_160k.pth'},
    #         checkpoint = [
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-0/fold-0_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-1/fold-1_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-2/fold-2_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-3/fold-3_0.pt',
    #             # '/home/r10user9/Documents/hhb/coatnet_baseline/result/mpvit_daformer_1024_2model_2stainnorm/fold-4/fold-4_0.pt'
    #         ],
    #     ),
    # ]
    # model = [
    #     dotdict(
    #         is_use = 1,
    #         # module = 'segformer.segformer',
    #         module = 'model_segformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/segformer.b2.512x512.ade.160k.pth'},
    #     ),
    # ]

    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_eff_unet',
    #         param={},
    #         checkpoint = [],
    #     ),
    # ]

    # model = [
    #     dotdict(
    #         is_use = 1,
    #         module = 'model_hornet_daformer',
    #         param={'encoder_ckpt': '/home/r10user9/Documents/hhb/coatnet_baseline/pretrain_model/upernet_hornet_tiny_gf.pth'},
    #         checkpoint = [
    #             '../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
    #         ],
    #     ),
    # ]

    net = [ load_net(m) for m in model if m.is_use==1 ][0][0]


    ## optimiser ----------------------------------
    if 0: ##freeze
        for p in net.stem.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                
    #freeze_bn(net)

    #-----------------------------------------------
    # import ipdb;ipdb.set_trace()
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, all_net[0][0].parameters()),lr=start_lr)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30, verbose=True)
    

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('\n')

    num_iteration = 300*len(train_loader)
    iter_log   = len(train_loader)*1 #479
    iter_valid = iter_log
    iter_save  = iter_log

    #%% Training
    log.write('** start training here! **\n')
    log.write('   batch_size = %d \n'%(batch_size))
    log.write('                     |---------------- VALID--------------|---- TRAIN/BATCH ----------------\n')
    log.write('rate     iter  epoch | dice   loss  best_dice  best_epoch | loss           | time           \n')
    log.write('-------------------------------------------------------------------------------------\n')

    def message(mode='print'):
        asterisk = ' '
        if mode==('print'):
            loss = batch_loss
        if mode==('log'):
            loss = train_loss
            if (iteration % iter_save == 0): asterisk = '*'

        text = \
            ('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
            '%4.3f  %4.3f  %4.4f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f   | '%(*valid_loss,) + \
            '%4.3f  %4.3f   | '%(*loss,) + \
            '%s' % ((time.time() - start_timer))

        return text

    valid_loss = np.zeros(4,np.float32)
    train_loss = np.zeros(2,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    start_timer = time.time()
    iteration = 0
    epoch = 0
    rate = 0

    # best_dice = 0
    # best_valepoch = 0
    # patience = 30
    while iteration < num_iteration:
        for t, batch in enumerate(train_loader):
            # import ipdb;ipdb.set_trace()
            # if iteration%iter_save==0:
            #     if iteration != 0:
            #         torch.save({
            #             'state_dict': net.state_dict(),
            #             'iteration': iteration,
            #             'epoch': epoch,
            #         }, out_dir + '/checkpoint/%08d.model.pth' %  (iteration))
            #         pass

            if (iteration%iter_valid==0):
                valid_loss = validate(net, valid_loader, epoch)
                if valid_loss[0] == best_dice:
                    torch.save(net.state_dict(), out_dir + f'/fold-{fold}_{epoch}.pt')
                pass

            if (iteration%iter_log==0) or (iteration%iter_valid==0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')


            # learning rate schduler ------------
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            # import ipdb;ipdb.set_trace()
            batch['image'] = batch['image'].half().cuda()
            batch['mask'] = batch['mask'].half().cuda()
            batch['organ'] = batch['organ'].cuda()
            # plt.imshow(batch['mask'][0])
            


            net.train()
            # net.output_type = ['loss']
            # if 1:
            # probability = 0
            # aux_loss = []
            with amp.autocast(enabled = is_amp):
                # for net in all_net:
                #     for n in net:
                net.output_type = ['loss']
                # import ipdb;ipdb.set_trace()
                output = net(batch)   # batch.shape: [8, 3, 768, 768], output.shape: [8, 1, 192, 192]
                # import ipdb;ipdb.set_trace()
                loss0  = output['bce_loss'].mean()
                # loss1  = output['aux2_loss'].mean()

            optimizer.zero_grad()
            # scaler.scale(loss0+0.2*loss1).backward()
            scaler.scale(loss0).backward()

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()


            # print statistics  --------
            # batch_loss[:2] = [loss0.item(),loss1.item()]
            batch_loss[:1] = [loss0.item()]
            sum_train_loss += batch_loss
            sum_train += 1
            if t % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            # print('\r', end='', flush=True)
            # print(message(mode='print'), end='', flush=True)
            # epoch += 1 / len(train_loader)
            # iteration += 1

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
            epoch += 1 / len(train_loader)
            iteration += 1
            # if epoch - best_valepoch > patience:
            #     break
            
        torch.cuda.empty_cache()
        
    log.write('\n')
    log.close()

if __name__ == '__main__':
	run(fold=0, patience=30, start_lr=5e-5, gpu=2, group_id=0, image_size=1024)