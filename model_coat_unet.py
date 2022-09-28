from kaggle_hubmap_kv3 import *
from daformer import *
from coat import *
import copy
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
# import lovasz_losses as L


#################################################################
def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit,mask)
    return loss


class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
	IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]
	
	def __init__(self, ):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
		self.register_buffer('std', torch.ones(1, 3, 1, 1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x - self.mean) / self.std
		return x


class Net(nn.Module):
	
	
	def __init__(self,
	             encoder=coat_lite_medium,
	             decoder=None,
	             encoder_cfg={},
	             decoder_cfg={},
				 encoder_ckpt=None,
				 decoder_ckpt=None
	             ):
		super(Net, self).__init__()
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		decoder_dim_unet = [256, 128, 64, 32, 16]
		
		# ----
		self.rgb = RGB()

		conv_dim = 32
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
		)
		self.convs = torch.nn.ModuleList()
		self.convs.append(self.conv)
		self.convs.append(copy.deepcopy(self.convs[0]))
		
		self.encoder_coat = encoder(
			#drop_path_rate=0.3,
		)
		if encoder_ckpt is not None:
			checkpoint = torch.load(encoder_ckpt, map_location=lambda storage, loc: storage)
			self.encoder_coat.load_state_dict(checkpoint['model'],strict=False)
		self.encoders_coat = torch.nn.ModuleList()
		self.encoders_coat.append(self.encoder_coat)
		self.encoders_coat.append(copy.deepcopy(self.encoders_coat[0]))
		encoder_dim = self.encoder_coat.embed_dims
		# [64, 128, 320, 512]
		
		# self.decoder_daformer = decoder(
		# 	encoder_dim=encoder_dim,
		# 	decoder_dim=decoder_dim,
		# )
		self.decoder_unet = UnetDecoder(
			encoder_channels=[0, conv_dim] + encoder_dim,
			decoder_channels=decoder_dim_unet,
			n_blocks=5,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)
		self.decoders_unet = torch.nn.ModuleList()
		self.decoders_unet.append(self.decoder_unet)
		self.decoders_unet.append(copy.deepcopy(self.decoders_unet[0]))

		# self.decoders = torch.nn.ModuleList()
		# for _ in range(5):
		# 	self.decoders.append(copy.deepcopy(self.decoder))

		self.logit_unet = nn.Sequential(
			nn.Conv2d(decoder_dim_unet[-1], 1, kernel_size=1),
		)
		self.output_type = ['inference', 'loss']
		# self.aux = nn.ModuleList([
        #     nn.Conv2d(decoder_dim, 1, kernel_size=1, padding=0) for i in range(4)
        # ])

		# self.encoder_decoders = torch.nn.ModuleList()
		# self.encoder_decoder = nn.Sequential(
		# 	self.encoder,
		# 	self.decoder
		# )
		# self.encoder_decoders.append(self.encoder_decoder)
		# self.encoder_decoders.append(copy.deepcopy(self.encoder_decoder))


	def forward(self, batch):
		# import ipdb;ipdb.set_trace()
		# mask = batch['mask']
		organs = batch['organ']
		x = batch['image']
		x = self.rgb(x)
		
		B, C, H, W = x.shape
		# import ipdb;ipdb.set_trace()
		# if self.training:
		# 	# encoder = []
		# 	# for i in range(B):
		# 	encoder = self.encoders[organs.item()](x)
		# 	# import ipdb;ipdb.set_trace()
		# 	# encoder = torch.concat(encoder)
		# else:
		encoder = self.encoders_coat[organs[0].item() // 4](x)
		conv = self.convs[organs[0].item() // 4](x)
		
		feature = encoder[::-1]
		head = feature[0]
		skip = feature[1:] + [conv, None]
		d = self.decoders_unet[organs[0].item() // 4].center(head)

		decoder = []
		for i, decoder_block in enumerate(self.decoders_unet[organs[0].item() // 4].blocks):
			s = skip[i]
			d = decoder_block(d, s)
			decoder.append(d)
		last = d

		# last, decoder = self.encoder_decoders[organs[0].item() // 4](x)
		# last, decoder = self.decoders_unet[organs[0].item() // 4](encoder)
		logit = self.logit_unet(last)
		# import ipdb;ipdb.set_trace()
		# print(logit.shape)
		# logit2 = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
		# mask = F.interpolate(mask, size=None, scale_factor=1/4, mode='bilinear', align_corners=False)
		
		output = {}
		# probability_from_logit = torch.sigmoid(logit)
		# output['probability'] = probability_from_logit
		# import ipdb;ipdb.set_trace()
		if 'loss' in self.output_type:
			# import ipdb;ipdb.set_trace()
			mask = batch['mask']
			# mask = F.interpolate(mask, size=None, scale_factor=1/4, mode='bilinear', align_corners=False)
			# output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, mask)
			# for i in range(4):
			# 	output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'])
		
		if 'inference' in self.output_type:
			output['probability'] = torch.sigmoid(logit)
		
		return output

 
 

 


def run_check_net():
	batch_size = 2
	image_size = 800
	
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	run_check_net()