import copy
import os
import pickle
import random
import shutil
import sys
import time
from shutil import get_terminal_size

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from progress.bar import Bar as Bar
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torchvision import transforms
from tqdm import tqdm

eps = 1e-7

class Recorder(object):
    def __init__(self, args):
        self.args = args

        self.test_acc_all = []
        self.test_acc_seen = []
        self.test_acc_novel = []

        self.algorithm = args.algorithm
        self.round = 0
        self.best_acc_all = 0
        self.mean_uncert = 0

    def validate(self, node):
        self.round += 1
        self.mean_uncert = test_foscr(self.args, node.test_all, node.algo, return_acc=False)

        all_cluster_results = test_cluster(self.args, node.test_all, node.algo, self.round)
        novel_cluster_results = test_cluster(self.args, node.test_novel, node.algo, self.round, offset=self.args.no_seen)
        test_acc_seen = test_seen(self.args, node.test_seen, node.algo, self.round)
        
        self.test_acc_all.append(round(all_cluster_results["acc"], 4)) 
        self.test_acc_seen.append(round(test_acc_seen, 4))
        self.test_acc_novel.append(round(novel_cluster_results['acc'], 4))
        
        if self.test_acc_all[-1] > self.best_acc_all:
            self.best_acc_all = self.test_acc_all[-1]

    def printer(self, node):



        print(f'test_acc_all: {self.test_acc_all}')
        print(f'test_acc_seen: {self.test_acc_seen}')
        print(f'test_acc_novel: {self.test_acc_novel}')

        save_path = f'saved_models/{self.algorithm}_{self.args.dataset}_{self.args.lbl_percent}_{self.args.novel_percent}_{self.args.run_started}_net_params.pth'
        torch.save(node.algo.model.state_dict(), save_path)
        proj_save_path = f'saved_models/{self.algorithm}_{self.args.dataset}_{self.args.lbl_percent}_{self.args.novel_percent}_{self.args.run_started}_proj_net_params.pth'
        torch.save(node.algo.proj_layer.state_dict(), proj_save_path)
        proto_save_path = f'saved_models/{self.algorithm}_{self.args.dataset}_{self.args.lbl_percent}_{self.args.novel_percent}_{self.args.run_started}_proto_net_params.pth'
        torch.save(node.algo.proto.data, proto_save_path)

    def finish(self):
        print('Finished!\n')
        print('Best Accuracy = {:.2f}%'.format(self.best_acc_all))





def Summary(args):
    print("Summary:\n")
    print("dataset:{}\tbatchsize:{}\n".format(args.dataset, args.batchsize))
    print("node_num:{},\tsplit:{}\n".format(args.node_num, args.split))
    print("global epochs:{},\tlocal epochs:{},\n".format(args.R, args.E))


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
 
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target):

    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save_path):
    filename=f'checkpoint.pth.tar'
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, f'model_best.pth.tar'))

class AverageMeter_nf(object):
    
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

def test(args, test_loader, model, return_acc=False):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(args.device), label.to(args.device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
         
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < args.no_seen

    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, mean_uncert {:.4f}'.format(overall_acc, seen_acc, unseen_acc, mean_uncert))
    
    if return_acc:
        return overall_acc, seen_acc, unseen_acc
    else:
        return mean_uncert

def test_foscr(args, test_loader, algo, return_acc=False):
    algo.model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    features = []

    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(args.device), label.to(args.device)
            ret_dict = algo.forward_cifar(x, None, label, evalmode=True)
            pred = ret_dict['label_pseudo']
            conf = ret_dict['conf']
            feat = ret_dict['features']

            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            features.append(feat.data.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)


    seen_mask = targets < args.no_seen
    unseen_mask = ~seen_mask

    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, mean_uncert {:.4f}'.format(overall_acc, seen_acc, unseen_acc, mean_uncert))
    if return_acc:
        return overall_acc, seen_acc, unseen_acc, mean_uncert
    else:
        return mean_uncert


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def test_seen(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            
        
            ret_dict = model.forward_cifar(inputs, None, targets, evalmode=True)
            outputs = ret_dict['logit']
            prec1, prec5 = accuracy_topk(outputs, targets, topk=(1, 5))

                
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.E,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                    # loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets =[]
    predictions = []
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            model.model.eval()
            ret_dict = model.forward_cifar(inputs, None, targets, evalmode=True)
            pred = ret_dict['label_pseudo']

            predictions.extend(pred.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.E,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)

    return eval_output

def plselect(node, args):
    
    # pseudo-label generation and selection
    node.model.zero_grad()
    lbl_unlbl_dict = pickle.load(open(f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl', 'rb'))
    total_samples = len(lbl_unlbl_dict['labeled_idx']) + len(lbl_unlbl_dict['unlabeled_idx'])
    # no_pl_perclass = int((args.pl_percent*total_samples)/(args.no_class*100))
    no_pl_perclass = int((args.lbl_percent*total_samples)/(args.all_class*100)/args.split)
    pl_dict, pl_acc, pl_no = pseudo_labeling(args, node.uncr_data, node.model, list(range(args.no_seen, args.all_class)), no_pl_perclass)
    with open(os.path.join(args.out, 'pseudo_labels_base.pkl'),"wb") as f:
        pickle.dump(pl_dict,f)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write(f'acc-pl: {pl_acc}, total-selected: {pl_no}\n')

@torch.no_grad()
def hungarian_evaluate(predictions, targets, offset=0):
    # Hungarian matching
    targets = targets - offset
    predictions = predictions - offset
    predictions_np = predictions.numpy()
    num_elems = targets.size(0)

    # only consider the valid predicts. rest are treated as misclassification
    valid_idx = np.where(predictions_np>=0)[0]
    predictions_sel = predictions[valid_idx]
    targets_sel = targets[valid_idx]
    num_classes = torch.unique(targets).numel()
    num_classes_pred = torch.unique(predictions_sel).numel()

    match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred, targets_k=num_classes) # match is data dependent
    reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions_sel == int(pred_i)] = int(target_i)

    # Gather performance metrics
    reordered_preds = reordered_preds.numpy()
    acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(num_elems) #accuracy is normalized with the total number of samples not only the valid ones
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())
    
    return {'acc': acc*100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

def pseudo_labeling(args, data_loader, model, novel_classes, no_pl_perclass):
    batch_time = AverageMeter()
    end = time.time()
    pseudo_idx = []
    pseudo_target = []
    pseudo_maxval = []
    gt_target = []
    model.eval()

    if not args.no_progress:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs, _) in enumerate(data_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            _, outputs = model(inputs)
            out_prob = F.softmax(outputs, dim=1)
            max_value, max_idx = torch.max(out_prob, dim=1)

            pseudo_target.extend(max_idx.cpu().numpy().tolist())
            pseudo_maxval.extend(max_value.cpu().numpy().tolist())
            pseudo_idx.extend(indexs.numpy().tolist())
            gt_target.extend(targets.cpu().numpy().tolist())

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                data_loader.set_description("pseudo-labeling itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    batch=batch_idx + 1,
                    iter=len(data_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            data_loader.close()

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_maxval = np.array(pseudo_maxval)
    pseudo_idx = np.array(pseudo_idx)

    #class balance the selected pseudo-labels
    blnc_idx_list = []
    for class_idx in novel_classes:
        current_class_idx = np.where(pseudo_target==class_idx)
        if len(np.where(pseudo_target==class_idx)[0]) > 0:
            current_class_maxval = pseudo_maxval[current_class_idx]
            sorted_idx = np.argsort(current_class_maxval)[::-1]
            current_class_idx = current_class_idx[0][sorted_idx[:no_pl_perclass]] 
            blnc_idx_list.extend(current_class_idx)

    if blnc_idx_list:
        blnc_idx_list = np.array(blnc_idx_list)
        pseudo_target = pseudo_target[blnc_idx_list]
        pseudo_idx = pseudo_idx[blnc_idx_list]
        gt_target = gt_target[blnc_idx_list]

    pl_eval_output = hungarian_evaluate(torch.from_numpy(pseudo_target), torch.from_numpy(gt_target), args.no_seen) # for sanity check only
    pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target':pseudo_target.tolist()}
 
    return pseudo_label_dict, pl_eval_output["acc"], len(pseudo_idx)

class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels.long(), self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class MarginLoss(nn.Module):
    
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)


class SimCLRTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    data_format is array or image
    """

    def __init__(self, size=32, gaussian=False, data_format="array"):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if gaussian:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    # torchvision.transforms.Resize(size=size),
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    # RandomApply(torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            if data_format == "array":
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(mode='RGB'),
                        # torchvision.transforms.Resize(size=size),
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ]
                )
            else:
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                    ]
                )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.fine_tune_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(mode='RGB'),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
        
def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def load__labels(label_dir):
    # image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
    labels = []
    with open(label_dir, 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()[1:])):
            fields = line.strip().split(',')[-1]
            labels.append(fields)
        labels = np.stack(labels, axis=0)
    return labels