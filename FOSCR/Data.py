
import math
import os
import os.path
import random

import medmnist
import numpy as np
import torch
from medmnist import INFO
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, Subset, random_split)
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


from randaugment import RandAugmentMC

# normalization parameters
cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
isic2018_mean, isic2018_std = (0.76369756, 0.54626846, 0.5707885), (0.08786482, 0.11671544, 0.13107239)
normal_mean, normal_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
mnist_mean, mnist_std = (0.1307), (0.3081)
medmnist_data = ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist",
        "bloodmnist", 'tissuemnist', "organamnist", "organcmnist", "organsmnist", "organmnist3d", "nodulemnist3d",
        "adrenalmnist3d", "fracturemnist3d", "vesselmnist3d", "synapsemnist3d"]

class Data(object):

    def __init__(self, args):
        self.args = args
        self.trainset, self.testset = None, None
        
        if args.dataset == "cifar10":
            args.all_class = 10
        elif args.dataset == "cifar100":
            args.all_class = 100
        elif args.dataset == "pathmnist":
            args.all_class = 9
        elif args.dataset == "bloodmnist":
            args.all_class = 8
        if args.dataset == "mnist":
            args.input_planes = 1

        args.no_seen = args.all_class - args.novel_percent
        args.proto_num = args.all_class
        self.merge_w = np.array([0 for _ in range(args.node_num)])

        dataset_class = get_dataset_class(args)

        train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel = dataset_class.get_dataset()

        # create dataloaders
        unlbl_batchsize = int((float(args.batchsize) * len(train_unlabeled_dataset))/(len(train_labeled_dataset) + len(train_unlabeled_dataset)))
        lbl_batchsize = args.batchsize - unlbl_batchsize
        args.iteration = (len(train_labeled_dataset) + len(train_unlabeled_dataset)) // args.batchsize // args.split

        if args.sampler == 'iid':
            
            num_train_labeled = [int(len(train_labeled_dataset) / args.split) for _ in range(args.split)] 
            num_train_labeled[-1] += len(train_labeled_dataset) % args.split
            splited_train_labeled_dataset = random_split(train_labeled_dataset, num_train_labeled, generator=torch.Generator().manual_seed(42))
            
            #
            num_train_unlabeled = [int(len(train_unlabeled_dataset) / args.split) for _ in range(args.split)]
            num_train_unlabeled[-1] += len(train_unlabeled_dataset) % args.split

            splited_train_unlabeled_dataset = random_split(train_unlabeled_dataset, num_train_unlabeled, generator=torch.Generator().manual_seed(42))

            args.merge_w = [1/args.split for _ in range(args.node_num)]
            

        elif args.sampler == "noniid-labeldir":

            targets = np.array(train_labeled_dataset.targets)
            targets_u = np.array(train_unlabeled_dataset.targets)

            N_l = len(train_labeled_dataset)
            N_u = len(train_unlabeled_dataset)
            N = N_l + N_u
            min_size = 0
            min_require_size = 10
            net_dataidx_map_l = {}
            net_dataidx_map_u = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(args.node_num)]
                idx_batch_u = [[] for _ in range(args.node_num)]

                for k in range(args.no_seen):
                    idx_k = np.where(targets == k)[0]
                    np.random.shuffle(idx_k)

                    proportions = np.random.dirichlet(np.repeat(args.noniid_beta, args.node_num))

                    ## Balance
                    proportions = np.array([p * (len(idx_j) < N / args.node_num) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
 
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    self.merge_w += [len(idx_j) for idx_j in idx_batch]
                

                for k in range(args.all_class):
                    idx_k = np.where(targets_u == k)[0]
                    np.random.shuffle(idx_k)
    
                    proportions = np.random.dirichlet(np.repeat(args.noniid_beta, args.node_num))

                    ## Balance
                    proportions = np.array([p * ((len(idx_j)+len(idx_j)) < N / args.node_num) for p, idx_j, idx_b in zip(proportions, idx_batch_u, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch_u = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_u, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch_u])
                    self.merge_w += [len(idx_j) for idx_j in idx_batch_u]

     

            for j in range(args.node_num):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map_l[j] = idx_batch[j]
            for j in range(args.node_num):
                np.random.shuffle(idx_batch_u[j])
                net_dataidx_map_u[j] = idx_batch_u[j]

            self.merge_w = self.merge_w / self.merge_w.sum()
            args.merge_w = self.merge_w
            splited_train_labeled_dataset = [Subset(train_labeled_dataset, net_dataidx_map_l[q]) for q in range(args.split)]
            
            splited_train_unlabeled_dataset = [Subset(train_unlabeled_dataset, net_dataidx_map_u[q]) for q in range(args.split)]
            print('merge-w:', self.merge_w)
        elif args.sampler == "iid-diff-quantity":
            idxs = np.random.permutation(int(len(train_labeled_dataset)))
            min_size = 0
            while min_size < 10:
                proportions = np.random.dirichlet(np.repeat(args.iid_beta, args.node_num))
                proportions = proportions/proportions.sum()
                min_size = np.min(proportions*len(idxs))
            proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
            batch_idxs = np.split(idxs,proportions)
            net_dataidx_map = {i: batch_idxs[i] for i in range(args.node_num)}

            for i in range(args.node_num):
                net_dataidx_map[i] = net_dataidx_map[i].tolist()
            


        self.test_all = DataLoader(test_dataset_all, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
        self.test_seen = DataLoader(test_dataset_seen, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
        self.test_novel = DataLoader(test_dataset_novel, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

        self.train_labeled_loader = [DataLoader(splited_train_labeled_dataset[i], batch_size=lbl_batchsize, shuffle=True, num_workers=args.num_workers) for i in range(args.node_num)]
    
        self.train_unlabeled_loader = [DataLoader(splited_train_unlabeled_dataset[i], batch_size=unlbl_batchsize, shuffle=True, num_workers=args.num_workers) for i in range(args.node_num)]
            
def get_dataset_class(args):
    if args.dataset == 'cifar10':
        return cifar10_dataset(args)
    elif args.dataset == "pathmnist":
        return pathmnist_dataset(args)
    elif args.dataset == "bloodmnist":
        return pathmnist_dataset(args)
    elif args.dataset == "mnist":
        return mnist_dataset(args)

def x_u_split_seen_novel(labels, lbl_percent, num_classes, lbl_set, unlbl_set, imb_factor):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        img_max = len(idx)
        num = img_max * ((1/imb_factor)**(i / (num_classes - 1.0)))
        idx = idx[:int(num)]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))

        if i in lbl_set: 
            labeled_idx.extend(idx[:n_lbl_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:])

        elif i in unlbl_set:
            unlabeled_idx.extend(idx)
    return labeled_idx, unlabeled_idx

class mnist_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
        ])
        self.transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
        ])

        base_dataset = datasets.MNIST(args.data_root, train=True, download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.all_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.all_class)), args.imb_factor)
        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.all_class = args.all_class
        self.algorithm = args.algorithm

    def get_dataset(self):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = MNISTSSL(self.data_root, train_labeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)
        train_unlabeled_dataset = MNISTSSL(self.data_root, train_unlabeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)
        
        test_dataset_seen = MNISTSSL_TEST(self.data_root, train=False, transform=self.transform_val, download=True, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = MNISTSSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(self.no_seen, self.all_class)))
        test_dataset_all = MNISTSSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel


class cifar10_dataset():
    def __init__(self, args):
        # ----------------------transforms----------------------------------

        self.transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        self.transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])


        #-------------------------------------------------------------------
        
            
        base_dataset = datasets.CIFAR10(args.data_root, train=True, download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.all_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.all_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.all_class = args.all_class
        self.algorithm = args.algorithm

    def get_dataset(self):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR10SSL(self.data_root, train_labeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)
        train_unlabeled_dataset = CIFAR10SSL(self.data_root, train_unlabeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)

        
        test_dataset_seen = CIFAR10SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = CIFAR10SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(self.no_seen, self.all_class)))
        test_dataset_all = CIFAR10SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel

class pathmnist_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),

        transforms.RandomGrayscale(p=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.transform_val = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
        ])

        data_flag = args.dataset
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])


        self.train_dataset = DataClass(split='train', transform=self.transform_train, download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(self.train_dataset.labels, args.lbl_percent, args.all_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.all_class)), args.imb_factor)
        # # balance the labeled and unlabeled data
        #     if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        #         exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        #         train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        #         if len(train_labeled_idxs) < len(train_unlabeled_idxs):
        #             diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
        #             train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        #         else:
        #             assert len(train_labeled_idxs) == len(train_unlabeled_idxs)
        
        self.test_dataset = DataClass(split='test', transform=self.transform_val, download=True)
        
        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.all_class = args.all_class
        self.algorithm = args.algorithm

    def get_dataset(self):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = pathmnistSSL(self.train_dataset, train_labeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)
        train_unlabeled_dataset = pathmnistSSL(self.train_dataset, train_unlabeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)#TransformTwice(self.transform_train)

        test_dataset_seen = pathmnistSSL_TEST(self.test_dataset, train=False, transform=self.transform_val, labeled_set=list(range(0,self.no_seen)), all_class=self.all_class)
        test_dataset_novel = pathmnistSSL_TEST(self.test_dataset, train=False, transform=self.transform_val, labeled_set=list(range(self.no_seen, self.all_class)), all_class=self.all_class)
        test_dataset_all = pathmnistSSL_TEST(self.test_dataset, train=False, transform=self.transform_val, all_class=self.all_class)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel

class pathmnistSSL(Dataset):
    def __init__(self, train_set, indexs, temperature=None, train=True,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        train_set.labels = np.array(train_set.labels.squeeze())
        if temperature is not None:
            self.temp = temperature*np.ones(len(train_set.labels))
        else:
            self.temp = np.ones(len(train_set.labels))
        
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = train_set.imgs[indexs]
            self.targets = np.array(train_set.labels)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.labels))


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.indexs[index], self.temp[index]


    def __len__(self):
        return len(self.data)

class pathmnistSSL_TEST(Dataset):
    def __init__(self, test_set, train=False,
                 transform=None, target_transform=None, labeled_set=None, all_class=10):
 
        self.transform = transform
        self.target_transform = target_transform
        test_set.labels = np.array(test_set.labels.squeeze())
        indexs = []
        self.data = test_set.imgs
        self.targets = test_set.labels
        if labeled_set is not None:
            for i in range(all_class):
                idx = np.where(test_set.labels == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)


            self.data = test_set.imgs[indexs]
            self.targets = np.array(test_set.labels)[indexs] 

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)   


    def __len__(self):
        return len(self.data)

class cifar100_dataset():
    def __init__(self, args):
        # augmentations
 
        self.transform_train = transforms.Compose([

            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])
        
        base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.all_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.all_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.all_class = args.all_class
        self.algorithm = args.algorithm
    def get_dataset(self):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR100SSL(self.data_root, train_labeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)

        train_unlabeled_dataset = CIFAR100SSL(self.data_root, train_unlabeled_idxs, train=True, transform=TransformTwice(self.transform_train), temperature=self.temperature)
   
  
        test_dataset_seen = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(self.no_seen, self.all_class)))
        test_dataset_all = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel

   
            

class generic128_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128, (0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
        ])
       #-------------------------------------------------------------------

        base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
        base_dataset_targets = np.array(base_dataset.imgs)
        base_dataset_targets = base_dataset_targets[:,1]
        base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
        if args.dataset == 'covid19':
            train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset_targets, args.lbl_percent, args.all_class, list(range(1, 4)), list(range(0, 1)), args.imb_factor)
        else:

            train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset_targets, args.lbl_percent, args.all_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.all_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.all_class = args.all_class
        self.algorithm = args.algorithm


    def get_dataset(self):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()
 
        train_labeled_dataset = GenericSSL(os.path.join(self.data_root, 'train'), train_labeled_idxs, transform=TransformTwice(self.transform_train), temperature=self.temperature)
        train_unlabeled_dataset = GenericSSL(os.path.join(self.data_root, 'train'), train_unlabeled_idxs, transform=TransformTwice(self.transform_train), temperature=self.temperature, temp_uncr=temp_uncr)

        test_dataset_seen = GenericTEST(os.path.join(self.data_root, 'test'), all_class=self.all_class, transform=self.transform_val, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = GenericTEST(os.path.join(self.data_root, 'test'), all_class=self.all_class, transform=self.transform_val, labeled_set=list(range(self.no_seen, self.all_class)))
        test_dataset_all = GenericTEST(os.path.join(self.data_root, 'test'), all_class=self.all_class, transform=self.transform_val)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel



class MNISTSSL(datasets.MNIST):

    def __init__(self, root, indexs, temperature=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.targets = np.array(self.targets)
        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class MNISTSSL_TEST(datasets.MNIST):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=True, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)


            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs] 

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, temperature=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.targets = np.array(self.targets)
        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))
        
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CIFAR10SSL_TEST(datasets.CIFAR10):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=True, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
  
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs] 

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, temperature=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))


        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL_TEST(datasets.CIFAR100):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(100):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class GenericSSL(datasets.ImageFolder):
    def __init__(self, root, indexs, temperature=None,
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))


        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class GenericTEST(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, labeled_set=None, all_class=200):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(all_class):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




#--------------- transforms ----------------
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
class TransformSW224:
    def __init__(self, transform, sz=224):
        self.transform = transform
        self.transform_strong = transforms.Compose([
            
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(sz),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=isic2018_mean, std=isic2018_std)
                
        ])
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform_strong(inp)
        return out1, out2

#--------------- preprocess ----------------

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p
    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p
    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        return ImageOps.equalize(img)

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
