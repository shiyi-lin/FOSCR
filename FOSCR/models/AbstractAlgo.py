from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter_nf

import models


class Algo(nn.Module):

    def __init__(self, model_type, args):
        super(Algo, self).__init__()
        self.args = args
        self.model = self.load_model(model_type)
        self.featdim = self.model.featdim
        self.bce = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.momem_proto = 0.9

        self.status = {"dense_sample_count": AverageMeter_nf()}

        self.normalizer = lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10

        # self.register_buffer("proto", self.normalizer(torch.randn((self.args.proto_num, 128), dtype=torch.float)).to(args.device))
        self.register_buffer("label_stat", torch.zeros(self.args.proto_num, dtype=torch.int))

        self.loss_stat_init()

    def loss_stat_init(self, ):
        self.bce_losses = AverageMeter_nf('bce_loss', ':.4e')
        self.ce_losses = AverageMeter_nf('ce_loss', ':.4e')
        self.entropy_losses = AverageMeter_nf('entropy_loss', ':.4e')
        self.cls_losses = AverageMeter_nf('cls_losses', ':.4e')
        self.cl_losses = AverageMeter_nf('cl_losses', ':.4e')
        self.feat_losses = AverageMeter_nf('feat_losses', ':.4e')
        self.disp_losses = AverageMeter_nf('disp_losses', ':.4e')
        self.simclr_losses = AverageMeter_nf('simclr_loss', ':.4e')
        self.supcon_losses = AverageMeter_nf('supcon_loss', ':.4e')
        self.semicon_losses = AverageMeter_nf('semicon_loss', ':.4e')
        self.ce_sup_losses = AverageMeter_nf('ce_sup_loss', ':.4e')
        self.proto_losses = AverageMeter_nf('proto_loss', ':.4e')




    def load_model(self, type="RN50_simclr"):
        model = None
        if type == "RN50_simclr":
            model = models.resnet50(num_classes=self.args.all_class)
            state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
            model.load_state_dict(state_dict, strict=False)

            for name, param in model.named_parameters():
                if 'fc' not in name and 'layer4' not in name:
                    param.requires_grad = False

            model = model.to(self.args.device)
            model.featdim = 2048

        if type == "RN18_simclr_CIFAR":
            model = models.resnet18(num_classes=self.args.all_class, in_channel=self.args.input_planes)
            model = model.to(self.args.device)
            if self.args.pretrained:
                if self.args.dataset == 'cifar10':
                    state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
                    model.load_state_dict(state_dict, strict=False)
                    # for name, param in model.named_parameters():
                    #     if 'linear' not in name and 'layer4' not in name:
                    #         param.requires_grad = False

                elif self.args.dataset == 'cifar100':
                    state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
                    model.load_state_dict(state_dict, strict=False)
                    # for name, param in model.named_parameters():
                    #     if 'linear' not in name and 'layer4' not in name:
                    #         param.requires_grad = False

            model.featdim = 512

            # for name, param in model.named_parameters():
            #     if 'linear' not in name and 'layer4' not in name:
            #         param.requires_grad = False
            model = model.to(self.args.device)

        return model

    @torch.no_grad()
    def update_label_stat(self, label):
        self.label_stat += label.bincount(minlength=self.args.proto_num).to(self.label_stat.device)

    @torch.no_grad()
    def reset_stat(self):
        self.label_stat = torch.zeros(self.args.proto_num, dtype=torch.int).to(self.label_stat.device)
        self.loss_stat_init()

    @torch.no_grad()
    def sync_prototype(self):
        pass

    def update_prototype_lazy(self, feat, label, weight=None, momemt=None):
        if momemt is None:
            momemt = self.momem_proto
        self.proto_tmp = self.proto.clone().detach()
        if weight is None:
            weight = torch.ones_like(label)
        for i, l in enumerate(label):
            alpha = 1 - (1. - momemt) * weight[i]
            self.proto_tmp[l] = self.normalizer(alpha * self.proto_tmp[l].data + (1. - alpha) * feat[i])

    def entropy(self, x):
        """
        Helper function to compute the entropy over the batch
        input: batch w/ shape [b, num_classes]
        output: entropy value [is ideally -log(num_classes)]
        """
        EPS = 1e-5
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)

        if len(b.size()) == 2:  # Sample-wise entropy
            return - b.sum(dim=1).mean()
        elif len(b.size()) == 1:  # Distribution-wise entropy
            return - b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))
