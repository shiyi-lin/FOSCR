import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='ssl')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--seen_device', type=str, default='0',
                        help='gpu id')
    parser.add_argument('--node_num', type=int, default=5,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=200,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=2, help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='', help='Notes of Experiments')
    parser.add_argument('--wandb', type=int, default='0',
                        help='wandb')
    parser.add_argument('--input_planes', type=int, default=3, help='input planes')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    
    

    # Data
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='./data, cifar10')
    parser.add_argument('--dataset', type=str, default='pathmnist',
                        help='datasets: {pathmnist, cifar10, mnist...}')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='batchsize')
    parser.add_argument('--split', type=int, default=5,
                        help='data split')
    parser.add_argument('--classes', type=int, default=10,
                        help='classes')
    parser.add_argument('--sampler', type=str, default='iid', help="iid, noniid-labeldir")
    parser.add_argument('--novel_percent', type=int, default='5', help="number of novel classes")
    parser.add_argument('--lbl_percent', type=int, default='10', help="labeled percentage")
    parser.add_argument('--imb_factor', type=int, default='1', help="imbalance factor")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--noniid_beta", default=0.8, type=float, help="dirchlet distribution")


    # Optima
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--stop_decay', type=int, default=150,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--wdecay', default=1e-4, type=float, help='weight decay')

    parser.add_argument('--description', default="default_run", type=str, help='description of the experiment')
    parser.add_argument('--out', default='outputs', help='Directory to output the result')
    parser.add_argument('--warmup-epochs', default=15, type=int, help='number of warmup epochs')
    parser.add_argument('--pretrained', action='store_true', help="pretrained resnet")
    parser.add_argument('--proto_align', action='store_true', help="proto_align")

    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--auto_scaler_target', default=0.7, type=float,
                        help='target weight for the first time scaling')
    parser.add_argument('--ema_update', action='store_true', help="pretrained resnet")
    parser.add_argument('--proto_ema_update', action='store_true', help="proto_ema_update")
    parser.add_argument('--ema_warmup_rounds', type=int, default=10, help='number of ema warmup rounds')


    #
    parser.add_argument('--w-semicon', type=float, default=0.1)
    parser.add_argument('--w-supcon', type=float, default=0.2)
    parser.add_argument('--w-supce', type=float, default=0.2)
    parser.add_argument('--w-proto', type=float, default=1.)
    parser.add_argument('--w-simclr', type=float, default=1.)
    parser.add_argument('--w-ent', default=0.05, type=float)
    parser.add_argument('--proto-num', default=10, type=int)
    parser.add_argument('--mask_policy', default="zuzuzoodzood", type=str)
    parser.add_argument('--id_thresh', type=int, default=70)
    parser.add_argument('--temp_simclr', default=.4, type=float)
    parser.add_argument('--temp_supcon', default=0.1, type=float)
    parser.add_argument('--temp_semicon', default=0.7, type=float)

    args = parser.parse_args()
    return args
