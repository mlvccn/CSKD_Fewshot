import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from clip import clip
from models.resnet18_encoder import *

feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

wp_dict = {
    'ensemble': [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    'itap': [
        "itap of a {}.",
    ],
    'origami': [
        "a origami {}.",
    ],
    'small': [
        "a photo of the small {}.",
    ],
    'class_name': [
        "{}",
    ],
    'lowres': [
        "a low resolution photo of the {}.",
    ],
}

def load_clip_to_cpu(backbone_name):
    
    url = clip._MODELS[backbone_name]
    backbone_name = backbone_name.replace('/', '-')
    model_path = os.path.join('/data/tianshaoqi24/cjj2/pretrain', backbone_name + '.pt')  # 加载本地模型
    # model_path = clip._download(url)    

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        templates = wp_dict['ensemble']
        with torch.no_grad():
            zeroshot_weights = []
            for classname in self.classnames:
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = self.clip_model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # L2 normalise text embedding
                class_embedding = class_embeddings.mean(dim=0) # take mean over all text embeddings for all prompts(均值)
                class_embedding /= class_embedding.norm() # L2 normalise mean embedding
                zeroshot_weights.append(class_embedding)
            # create shape NxC
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        return zeroshot_weights
    
    def forward_old(self):
        temp = 'a photo of {}, a type of bird'
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x

class CustomCLIP(nn.Module):

    def __init__(self, classnames, clip_model, args):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.args = args
        if args.base_mode not in feat_dims:
            args.base_mode = 'ViT-B/16'

        emb_dim = feat_dims[args.base_mode] # 根据传入的网络结构决定嵌入向量的维度
        self.adapter = Adapter(emb_dim, 4).to(clip_model.dtype)
        # self.cache_proto = None     # 存储视觉原型

    def encode_text(self):
        if self.args.dataset == 'cub200':
            self.text_features = self.text_encoder.forward_old()
        else:
            self.text_features = self.text_encoder()    # 固定值 只需计算一次

    def forward(self, image, class_num):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2     # 比例系数
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_features[: class_num]
        # text_features = self.text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits