# import new Network name here and add in model_class args
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import random

import numpy as np
from losses import SupContrastive



def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()     # 损失  
    ta = Averager()     # 准确率

    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, single_labels = batch
        b, c, h, w = data.shape
        original = data.cuda(non_blocking=True)
        single_labels = single_labels.cuda(non_blocking=True)
        
        data_classify = original     # (64, 3, 224, 224)
        
        preds = model(data_classify, args.base_class)
        # preds = model(data_classify, 200)
        total_loss = F.cross_entropy(preds, single_labels)
        
        acc = count_acc(preds, single_labels)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()

    return tl, ta


def replace_base_fc(trainset, test_transform, model, args):
    # replace fc.weight with the embedding average of train data(利用原型替换线性分类器权重参数)
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, labels = [_.cuda() for _ in batch]
            b = data.size()[0]
           
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
            
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class] = proto_list

    return model


def update_fc_ft(trainloader, model, session, args, testloader, base_trainloader):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session 



    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    elif args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr_new}],
                                      weight_decay=0)


    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, labels = [_.cuda() for _ in batch]
                b, c, h, w = data.shape
                preds = model(data, new_class)

                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print(f"session {session} fine tune epoch {epoch}:  train loss = {loss.item()}")

            test(model, testloader, epoch, args, session)

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]

            preds = model(data, test_class)
            
            loss = F.cross_entropy(preds, test_label)
            acc = count_acc(preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('session {}, epoch {}, test, loss={:.4f} acc={:.4f}'.format(session, epoch, vl, va))

    return vl,va


def test_emb(model, testloader, epoch, args, session):
    
    emb_list = []
    label_list = []
    
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]

            emb, preds = model(data, test_class)
            
            emb_list.append(emb)
            label_list.append(test_label)

            loss = F.cross_entropy(preds, test_label)
            acc = count_acc(preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('session {}, epoch {}, test, loss={:.4f} acc={:.4f}'.format(session, epoch, vl, va))

    emb_array = torch.cat(emb_list).cpu().numpy()
    label_array = torch.cat(label_list).cpu().numpy()

    print("emb_array dim: ", emb_array.shape)
    print("label_array dim: ", label_array.shape)

    np.save('save_array/clip_emb.npy', emb_array)
    np.save('save_array/clip_emb_label.npy', label_array)

    return vl,va