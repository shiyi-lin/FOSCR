import time
from itertools import cycle

import torch
from utils import AverageMeter, Bar
import torch.nn as nn


bce = nn.BCELoss()

def train_foscr(node, args, m):
    node.algo.model.train()
    node.algo.proj_layer.train()
    node.algo.m = -min(m, 0.5)
    batch_time = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=len(node.train_unlabeled_data))   
    
    label_loader_iter = cycle(node.train_labeled_data)

    for batch_idx, ((ux, ux2), target_unlabeled) in enumerate(node.train_unlabeled_data):

        ((x, x2), target) = next(label_loader_iter)
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)

        x, x2, target, target_unlabeled = x.to(args.device), x2.to(args.device), target.to(args.device), target_unlabeled.to(args.device)
        node.optimizer.zero_grad()
        loss = node.algo.forward_cifar(x, x2, target, target_unlabeled=target_unlabeled)
        loss.backward()

        node.optimizer.step()
        node.algo.sync_prototype()
        end2 = time.time()

        batch_time.update(end2 - end)
        
        bar.suffix  = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | Loss: {losses:.4f} '.format(
                    batch=batch_idx+1,
                    size=len(node.train_unlabeled_data),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    losses = node.algo.losses.avg,

                    )
        bar.next()
    bar.finish()

class Trainer(object):

    def __init__(self, args):
        self.train = train_foscr
    def __call__(self, node, args, m):
        self.train(node, args, m)



