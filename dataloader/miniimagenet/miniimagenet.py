import os
import os.path as osp



import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

from .autoaugment_mini import AutoAugImageNetPolicy

import json

class MiniImageNet(Dataset):

    def __init__(self, root='/data/tianshaoqi24/datasets', train=True, index_path=None, index=None,
                 base_sess=None, train_transform=None, test_transform=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = None
        self.crop_transform = train_transform
        self.secondary_transform = test_transform
        if isinstance(test_transform, list):
            assert(len(test_transform) == self.crop_transform.N_large + self.crop_transform.N_small)
        self.multi_train = False  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'mini-imagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'mini-imagenet/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        self.class_name = []        # 图像类别对应的字符串名称
        lb = -1

        with open(os.path.join(root, 'mini-imagenet/classname.json'), 'r') as f:
            class2name = json.load(f)

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.class_name.append(class2name[wnid][0])

            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        
        # print(self.class_name)

        if train:
            image_size = 224
            # self.transform = transforms.Compose([
            #         transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            #         transforms.RandomHorizontalFlip(p=0.5),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            #     ])
            self.transform = train_transform

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 224
            self.transform = transforms.Compose([
                # transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.transform = test_transform
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        total_image = self.transform(Image.open(path).convert('RGB'))
        return total_image, targets
