import copy
import torch

def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer

def exclude_bias_and_norm(p):
    return p.ndim == 1

def build_foscr_model(args):
    from models.OpenSupCon import OpenSupCon
    algo = OpenSupCon(args.arch, args)
    return algo

class Node(object):
    def __init__(self, num, train_labeled_data,train_unlabeled_data, test_all, test_seen, test_novel, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_labeled_data = train_labeled_data
        self.train_unlabeled_data = train_unlabeled_data
        
        self.test_all = test_all
        self.test_seen = test_seen
        self.test_novel = test_novel
        self.weight_scaler = None
        self.count = 0

        self.algo = build_foscr_model(args)
        self.optimizer = init_optimizer(self.algo, self.args)

        self.losses = []
    def _calculate_divergence(self, old_model, new_model, typ='l2'):
        size = 0
        total_distance = 0
        old_dict = old_model.state_dict()
        new_dict = new_model.state_dict()
        for name in old_dict.keys():
            if 'conv' in name and 'weight' in name:
                total_distance += self._calculate_distance(old_dict[name].detach().clone().view(1, -1),
                                                           new_dict[name].detach().clone().view(1, -1),
                                                           typ)
                size += 1
        distance = total_distance / size
     
        return distance
    def _calculate_distance(self, m1, m2, typ='l2'):
        if typ == 'l2':
            return torch.dist(m1, m2, 2)
    def fork(self, global_node):

        self.algo = copy.deepcopy(global_node.algo)
        self.optimizer = init_optimizer(self.algo.model, self.args)



class Global_Node(object):
    def __init__(self, test_all, test_seen, test_novel, args):
        self.num = 0
        self.args = args
        self.device = self.args.device

        self.algo = build_foscr_model(args)
        self.model = self.algo.model

        self.test_all = test_all
        self.test_seen = test_seen
        self.test_novel = test_novel
        self.init = False
        self.save = []
        self.losses = []

    def merge(self, Edge_nodes):
     
        Node_State_List_model = [copy.deepcopy(Edge_nodes[i].algo.model.state_dict()) for i in range(len(Edge_nodes))]
        self.Dict_model = Node_State_List_model[0]
        for key in self.Dict_model.keys():
            self.Dict_model[key] = self.args.merge_w[0] * self.Dict_model[key]

        for key in self.Dict_model.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict_model[key] += self.args.merge_w[i]*Node_State_List_model[i][key]

            self.Dict_model[key] = self.Dict_model[key].float()     
   
        self.algo.model.load_state_dict(self.Dict_model)

        Node_State_List_model = [copy.deepcopy(Edge_nodes[i].algo.proj_layer.state_dict()) for i in range(len(Edge_nodes))]
        self.Dict_model = Node_State_List_model[0]

        for key in self.Dict_model.keys():
            self.Dict_model[key] = self.args.merge_w[0] * self.Dict_model[key]
        for key in self.Dict_model.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict_model[key] += self.args.merge_w[i]*Node_State_List_model[i][key]

            self.Dict_model[key] = self.Dict_model[key].float()     

        self.algo.proj_layer.load_state_dict(self.Dict_model)

        self.algo.proto.data = self.args.merge_w[0]*Edge_nodes[0].algo.proto.data

        for i in range(1, len(Edge_nodes)):
            self.algo.proto.data += self.args.merge_w[i]*Edge_nodes[i].algo.proto.data

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):

    params = current_model.state_dict()
    ema_params = ma_model.state_dict()
    for param, ema_param in zip(params.values(), ema_params.values()):
        old_weight, up_weight = ema_param.data, param.data
        ema_param.data = ema_updater.update_average(old_weight, up_weight)
    ma_model.load_state_dict(ema_params)
    
        
