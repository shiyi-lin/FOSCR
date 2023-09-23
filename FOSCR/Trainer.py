import time
from itertools import cycle

import torch
import torch.nn.functional as F
from utils import AverageMeter, Bar



def train_opencon(node, args):
    node.algo.model.train()
    node.algo.proj_layer.train()

    batch_time = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.iteration)   
    
    label_loader_iter = cycle(node.train_labeled_data)

    for batch_idx, ((ux, ux2), target_unlabeled, _, _) in enumerate(node.train_ublabeled_data):

        ((x, x2), target, _, _) = next(label_loader_iter)
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
        
    # plot progress
        bar.suffix  = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | Loss_simclr: {losses_simclr:.4f} | Loss_supcon: {losses_supcon:.4f} | Loss_semicon: {losses_semicon:.4f} | Loss_entrop: {losses_ent:.4f} | Loss_ce: {losses_ce:.4f} | Loss_proto: {losses_proto:.4f}'.format(
                    batch=batch_idx,
                    size=args.iteration,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,

                    losses_simclr = node.algo.simclr_losses.avg,
                    losses_supcon = node.algo.supcon_losses.avg,
                    losses_semicon = node.algo.semicon_losses.avg,
                    losses_ent=node.algo.entropy_losses.avg,
                    losses_ce=node.algo.ce_sup_losses.avg,
                    losses_proto=node.algo.proto_losses.avg,
                    )
        bar.next()
    bar.finish()

class Trainer(object):

    def __init__(self, args):
 
        self.train = train_opencon
    def __call__(self, node, args):

        self.train(node, args)



