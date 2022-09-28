import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
from . import mix_transformer

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
    def __init__(self, backbone='mit_b4', num_classes=1, embedding_dim=512, pretrained=None, encoder_ckpt=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        # import ipdb;ipdb.set_trace()
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            # state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict = torch.load(encoder_ckpt, map_location=lambda storage, loc: storage)
            # state_dict.pop('head.weight')
            # state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, strict=False)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.logit = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

        # self.logit = nn.Sequential(
		# 	nn.Conv2d(decoder_dim, 1, kernel_size=1),
		# )
        self.output_type = ['inference', 'loss']
        self.rgb = RGB()

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.logit.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.logit.weight)

        return param_groups

    def forward(self, batch):
        
        x = batch['image']
        x = self.rgb(x)
        # import ipdb;ipdb.set_trace()
        encoder = self.encoder(x)
        # _x1, _x2, _x3, _x4 = _x
        # cls = self.classifier(_x4)
        decoder = self.decoder(encoder)
        # import ipdb;ipdb.set_trace()
        # logit = self.logit(decoder[-1])
        logit = decoder
        logit2 = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        # import ipdb;ipdb.set_trace()
        output = {}
        if 'loss' in self.output_type:
            # import ipdb;ipdb.set_trace()
            mask = batch['mask']
            mask = F.interpolate(mask, size=None, scale_factor=1/4, mode='bilinear', align_corners=False)
			# output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, mask)
            # for i in range(4):
            #     output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'])
        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit2)
        # import ipdb;ipdb.set_trace()

        return output