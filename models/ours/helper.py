# import new Network name here and add in model_class args
from .Network import CSKD
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from losses import SupContrastive


def base_train(model, trainloader, optimizer, scheduler, epoch, args, clip_model):
    tl = Averager()
    ta = Averager()
    model = model.train()
    model.mode = 'cos'
    tqdm_gen = tqdm(trainloader)
    clip_model.eval()
    torch.autograd.set_detect_anomaly(True)
    for i, batch in enumerate(tqdm_gen, 1):
        data, labels = [_.cuda() for _ in batch]
        b, c, h, w = data.shape
        
        embs, logits = model.forward_metric(data)
        with torch.no_grad():
            clip_preds = clip_model(data, args.base_class)

        gt_loss = F.cross_entropy(logits, labels)       # FC层的预测

        ##
        # KL_loss = 0
        KL_loss = F.kl_div(logits.log_softmax(dim=-1).to(torch.float32), clip_preds.softmax(dim=-1).to(torch.float32), reduction='batchmean')
            
        acc = count_acc(logits, labels)

        total_loss = KL_loss +  gt_loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
                'Session {}, epo {}, lrc={:.4f},total loss={:.4f}, gt_loss = {:.4f}, kl loss = {:.4f}  acc={:.4f}'.format(0, epoch, 
                                                                                                                          lrc, 
                                                                                                                          total_loss.item(), 
                                                                                                                          gt_loss.item(),
                                                                                                                          KL_loss.item(),
                                                                                                                          acc))
        tl.add(total_loss.item())

        ##
        # total_loss = KL_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, test_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:] = proto_list
    model.mode = 'cos'
    return model


def finetune(model, trainloader, optimizer, scheduler, session, args, clip_model):
    tl = Averager()
    ta = Averager()
    model = model.train()
    model.mode = 'encoder'
    tqdm_gen = tqdm(trainloader)
    
    
    for epoch_ in range(args.epochs_new):
        for i, batch in enumerate(tqdm_gen, 1):
            data, labels = [_.cuda() for _ in batch]
            b, c, h, w = data.shape
            
            embs = model(im_cla=data)
            clip_preds = clip_model(data, args.base_class)


            # text_features = clip_model.text_features[:args.base_class]
            text_features = clip_model.text_features
            image_features = embs / embs.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = clip_model.logit_scale.exp()
            preds = logit_scale * image_features.float() @ text_features.t().float()
            # preds = joint_preds[:, :args.base_class*m]    # [768, 1200]
            gt_loss = F.cross_entropy(preds, labels)

            ##
            KL_loss = F.kl_div(preds.log_softmax(dim=-1), clip_preds.softmax(dim=-1), reduction='batchmean')
            
            acc = count_acc(preds, labels)

            total_loss = KL_loss + 0 * gt_loss

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session {}, epo {}, lrc={:.4f},total loss={:.4f}, gt_loss = {:.4f}, kl loss = {:.4f}  acc={:.4f}'.format(session, epoch_, 
                                                                                                                          lrc, 
                                                                                                                          total_loss.item(), 
                                                                                                                          gt_loss.item(),
                                                                                                                          KL_loss.item(),
                                                                                                                          acc))
            tl.add(total_loss.item())
            ta.add(acc)

            ##

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def update_fc_ft(trainloader, model, session, args, clip_model):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session 
    new_fc = nn.Parameter(
        torch.rand(args.way, model.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class : new_class, :].data)

    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                    #  {'params': model.encoder_q.fc.parameters(), 'lr': 0.05*args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001*args.lr_new}
                                     ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                      {'params': model.encoder_q.fc.parameters(), 'lr': 0.01*args.lr_new},
                                      {'params': model.encoder_q.layer3.parameters(), 'lr':0.02*args.lr_new}],
                                      weight_decay=0)
        
    # criterion = SupContrastive().cuda() 

    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, labels = [_.cuda() for _ in batch]
                b, c, h, w = data.shape
                
            old_fc = model.fc.weight[:old_class, :].clone().detach() 

            clip_preds = clip_model(data, new_class)

            fc = torch.cat([old_fc, new_fc], dim=0)
            features, _ = model.encode_q(data)
            features.detach()
            
            logits = model.get_logits(features, fc)
            loss = F.cross_entropy(logits, labels)

            KL_loss = F.kl_div(logits.log_softmax(dim=-1), clip_preds.softmax(dim=-1), reduction='batchmean')
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

    model.fc.weight.data[old_class : new_class, :].copy_(new_fc.data)

def test(model, testloader, epoch, args, session, clip_model):
    test_class = args.base_class + session * args.way
    model.mode = 'cos'
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            embs, preds = model.forward_metric(data)
            
            loss = F.cross_entropy(preds, test_label)
            acc = count_acc(preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl,va