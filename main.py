import os
from datetime import datetime
import torch
from Args import args_parser
from Data import Data as Data
from Node import Global_Node, Node
from Trainer import Trainer
from utils import (Recorder, Summary, seed_torch)
args = args_parser()

args.device = torch.device('cuda:'+args.seen_device if torch.cuda.is_available() else 'cpu')
args.use_cuda = torch.cuda.is_available()
args.split = args.node_num


run_started = datetime.today().strftime('%d-%m-%y_%H%M%S')
args.exp_name = f'dataset_{args.dataset}_algo_{args.algorithm}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_node_num_{args.node_num}_E_{args.E}_bs_{args.batchsize}_sampler_{args.sampler}'
args.out = os.path.join(args.out, args.exp_name)
args.run_started = run_started
os.makedirs(args.out, exist_ok=True)
with open(f'{args.out}/parameters.txt', 'a+') as ofile:
    ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))

print('Running on', args.seen_device)

seed_torch()

Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
Summary(args)
name = f'{args.dataset}_{args.algorithm}_{args.batchsize}_sampler_{args.sampler}_wce_{args.w_supce}_wsup_{args.w_supcon}_wsemi_{args.w_semicon}'
print('name:', name)

# init nodes
Global_node = Global_Node(Data.test_all, Data.test_seen, Data.test_novel, args)
Edge_nodes = [Node(k, Data.train_labeled_loader[k], Data.train_unlabeled_loader[k], Data.test_all, Data.test_seen, Data.test_novel, args) for k in range(args.node_num)]

for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    if rounds in [100, 150]:
        args.lr *= args.schw
    for k in range(len(Edge_nodes)):
        print(f'----------No. {rounds+1} rounds No. {k+1} node---------------')
        
        Edge_nodes[k].fork(Global_node)   
        print('Learning rate={:.4f}'.format(Edge_nodes[k].optimizer.param_groups[0]['lr']))
        
        for epoch in range(args.E):

            Edge_nodes[k].algo.reset_stat()
            Train(Edge_nodes[k], args, recorder.mean_uncert)


    Global_node.merge(Edge_nodes)        
    Global_node.algo.count += 1    
    
    recorder.validate(Global_node)
    print('===============The {:d}-th round==============='.format(rounds + 1))
    recorder.printer(Global_node)

recorder.finish()
Summary(args)

