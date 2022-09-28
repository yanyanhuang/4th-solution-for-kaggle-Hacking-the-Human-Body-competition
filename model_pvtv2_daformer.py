from kaggle_hubmap_kv3 import *
from daformer import *
from pvt_v2 import *
import copy



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
	             encoder=pvt_v2_b4,
	             decoder=daformer_conv3x3,
	             encoder_cfg=dict(img_size=1024),
	             decoder_cfg={},
				 encoder_ckpt=None,
				 decoder_ckpt=None
	             ):
		super(Net, self).__init__()
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		
		# ----
		self.rgb = RGB()
		
		# self.encoder = encoder(
		# 	#drop_path_rate=0.3,
		# )
		self.encoder = encoder(
			**encoder_cfg
		)
		if encoder_ckpt is not None:
			checkpoint = torch.load(encoder_ckpt, map_location=lambda storage, loc: storage)
			# import ipdb;ipdb.set_trace()
			# self.encoder.load_state_dict({k.replace('backbone.', ''):v for k, v in checkpoint['state_dict'].items()}, strict=False)
			self.encoder.load_state_dict(checkpoint,strict=False)
		self.encoders = torch.nn.ModuleList()
		# for _ in range(2):
		self.encoders.append(self.encoder)
		self.encoders.append(copy.deepcopy(self.encoders[0]))
		encoder_dim = self.encoder.embed_dims
		# encoder_dim = [64, 128, 256, 512]
		# [64, 128, 320, 512]
		
		self.decoder = decoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
		)
		# self.decoders = torch.nn.ModuleList()
		# self.decoders.append(self.decoder)
		# self.decoders.append(copy.deepcopy(self.decoders[0]))

		# self.decoders = torch.nn.ModuleList()
		# for _ in range(5):
		# 	self.decoders.append(copy.deepcopy(self.decoder))

		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim, 1, kernel_size=1),
		)
		self.output_type = ['inference', 'loss']
		# self.aux = nn.ModuleList([
        #     nn.Conv2d(decoder_dim, 1, kernel_size=1, padding=0) for i in range(4)
        # ])

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
		encoder = self.encoders[organs[0].item() // 4](x)
		#print([f.shape for f in encoder])
		import ipdb;ipdb.set_trace()

		last, decoder = self.decoder(encoder)
		# last, decoder = self.decoders[organs[0].item()](encoder)
		logit = self.logit(last)
		# import ipdb;ipdb.set_trace()
		# print(logit.shape)
		logit2 = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
		# mask = F.interpolate(mask, size=None, scale_factor=1/4, mode='bilinear', align_corners=False)
		
		output = {}
		# probability_from_logit = torch.sigmoid(logit)
		# output['probability'] = probability_from_logit
		# import ipdb;ipdb.set_trace()
		if 'loss' in self.output_type:
			# import ipdb;ipdb.set_trace()
			mask = batch['mask']
			mask = F.interpolate(mask, size=None, scale_factor=1/4, mode='bilinear', align_corners=False)
			# output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, mask)
			# for i in range(4):
			# 	output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'])
		
		if 'inference' in self.output_type:
			output['probability'] = torch.sigmoid(logit2)
		
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