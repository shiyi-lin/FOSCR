import os

from datetime import datetime

# import numpy as np
import torch

import wandb
from Args import args_parser
from Data import Data
from Node import Global_Node, Node

from Trainer import Trainer
from utils import (LR_scheduler, Recorder, Summary,
                 seed_torch)

args = args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.seen_device

args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.use_cuda = torch.cuda.is_available()
args.split = args.node_num

args.epochs= args.E

args.data_root = os.path.join(args.data_root, args.dataset)
os.makedirs(args.data_root, exist_ok=True)
run_started = datetime.today().strftime('%d-%m-%y_%H%M%S')
args.exp_name = f'dataset_{args.dataset}_algo_{args.algorithm}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_{args.description}_{run_started}'
args.out = os.path.join(args.out, args.exp_name)
args.run_started = run_started
os.makedirs(args.out, exist_ok=True)
with open(f'{args.out}/parameters.txt', 'a+') as ofile:
    ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
best_acc = 0 
print('Running on', args.seen_device)

seed_torch()

Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
Summary(args)
# logs
if args.wandb:
    wandb.init(project="NCD_FL_v2", entity="yii")  
    config = wandb.config
    config.communications_round = args.R
    config.node_num = args.node_num
    config.dataset = args.dataset
    config.local_epoch = args.E

    config.lbl_percent = args.lbl_percent
    config.novel_percent = args.novel_percent
    config.imb_factor = args.imb_factor
 
    config.lr = args.lr
    config.algorithm = args.algorithm
    config.pretrained = args.pretrained

    config.w_supce = args.w_supce
    config.w_supcon = args.w_supcon
    config.w_semicon = args.w_semicon
    config.w_simclr = args.w_simclr
    config.sampler = args.sampler
    config.proto_align = args.proto_align
    config.w_proto = args.w_proto
    config.noniid_beta = args.noniid_beta


# init nodes
Global_node = Global_Node(Data.test_all, Data.test_seen, Data.test_novel, args)
Edge_nodes = [Node(k, Data.train_labeled_loader[k], Data.train_unlabeled_loader[k], Data.test_all, Data.test_seen, Data.test_novel, args) for k in range(args.node_num)]

# train
for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))

    # LR_scheduler(rounds, Edge_nodes, args)

    
    for k in range(len(Edge_nodes)):
        print(f'----------No. {rounds+1} rounds No. {k+1} node---------------')
        Edge_nodes[k].fork(Global_node)   
        print('Learning rate={:.4f}'.format(Edge_nodes[k].optimizer.param_groups[0]['lr']))
        
        for epoch in range(args.E):
    
            Edge_nodes[k].algo.reset_stat()
            Train(Edge_nodes[k], args)
            # Edge_nodes[k].scheduler.step()


    Global_node.merge(Edge_nodes)            
    
    recorder.validate(Global_node)
    print('===============The {:d}-th round==============='.format(rounds + 1))
    recorder.printer(Global_node)
    if args.wandb:
        wandb.log({"test_acc_all": recorder.test_acc_all[str(Global_node.num)][rounds]})
        wandb.log({"test_acc_seen": recorder.test_acc_seen[str(Global_node.num)][rounds]})
        wandb.log({"test_acc_novel": recorder.test_acc_novel[str(Global_node.num)][rounds]})
recorder.finish()
Summary(args)





