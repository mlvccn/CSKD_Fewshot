import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.sampler import CategoriesSampler
from augmentations.constrained_cropping import CustomMultiCropDataset, CustomMultiCropping

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):

    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, index=class_index, base_sess=True,
                                          crop_transform=args.tfm_train, secondary_transform=None)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,index=class_index, base_sess=True,  crop_transform=args.tfm_test)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,index=class_index, base_sess=True,
                                       crop_transform=args.tfm_train, secondary_transform=None)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index, crop_transform=args.tfm_test)

    if args.dataset == 'mini_imagenet':
        ##########################################
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                             train_transform=args.tfm_train)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, test_transform=args.tfm_test)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args,session):
    # crop_transform, secondary_transform = get_transform(args)
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False,
                                         crop_transform=args.tfm_train, secondary_transform=None)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path,
                                       crop_transform=args.tfm_test, secondary_transform=None)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path,
                                             train_transform=args.tfm_train)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False, crop_transform=args.tfm_test)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new, crop_transform=args.tfm_train)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new, test_transform=args.tfm_test)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list