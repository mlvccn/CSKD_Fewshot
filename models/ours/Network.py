import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet18_encoder import *
from models.resnet20_cifar import *


class CSKD(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder_q = resnet20(num_classes=128)
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder_q = resnet18(False, args, num_classes=128)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder_q = resnet18(True, args, num_classes=128) 
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.base_class, bias=False)
        

    def forward_metric(self, x):
        x, _ = self.encode_q(x)
        logits = None
        if 'cos' in self.mode:
            logits = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            logits = self.args.temperature * logits
        elif 'dot' in self.mode:
            x = self.fc(x)
        return x, logits # joint, contrastive

    def encode_q(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def forward(self, im_cla, base_sess=True, last_epochs_new=False):
        if self.mode != 'encoder':
            x = self.forward_metric(im_cla)
            return x 
        elif self.mode == 'encoder':
            x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')
    
    
    def update_fc(self,dataloader,class_list, session, clip_model=None, args=None):
        for batch in dataloader:                                # 一次batch可以将 N-way K-shot样本取出
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            # data = transform(data)
            # m = data.size()[0] // b 
            # labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            data, _ =self.encode_q(data)
            data.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list, clip_model, args)
            self.fc.weight.data
            self.fc.weight.data = torch.cat([self.fc.weight.data, new_fc], dim=0)    

    def update_fc_avg(self,data,labels,class_list, clip_model=None, args=None):
        new_fc=[]
        for class_index in class_list:
            data_index=(labels==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
        new_fc=torch.stack(new_fc,dim=0)

        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))