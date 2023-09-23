import copy
import torch
from torch import optim

import Model

def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer

def exclude_bias_and_norm(p):
    return p.ndim == 1

class Node(object):
    def __init__(self, num, train_labeled_data,train_ublabeled_data, test_all, test_seen, test_novel, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_labeled_data = train_labeled_data
        self.train_ublabeled_data = train_ublabeled_data
        
        self.test_all = test_all
        self.test_seen = test_seen
        self.test_novel = test_novel
        self.weight_scaler = None
        self.count = 0

    
        self.algo = Model.build_opencon_model(args)
        self.optimizer = init_optimizer(self.algo.model, self.args)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1)
        
      
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

        if self.args.ema_update:
            self.count += 1
            if self.count > self.args.ema_warmup_rounds:
                if self.args.proto_ema_update:
                    self.encoder_distance = torch.dist(self.algo.proto.data, global_node.algo.proto.data, 2)
                else:
                    self.encoder_distance = self._calculate_divergence(self.algo.model, global_node.algo.model)
                if self.encoder_distance == 0:
                    weight = 0
                else:
                    if not self.weight_scaler:
                        self.weight_scaler = self.args.auto_scaler_target / self.encoder_distance

                    weight = self.encoder_distance
                    weight = min(1, self.weight_scaler * weight)
                    weight = 1 - weight
            else:
                weight = 0

            print('weight:', weight)
         
          
            Node_State_model = copy.deepcopy(self.algo.model.state_dict())
            Global_State_model = copy.deepcopy(global_node.algo.model.state_dict())
        

            for key in Node_State_model.keys():
                
                Node_State_model[key] = weight * Node_State_model[key] + (1-weight)*Global_State_model[key]

            self.algo.model.load_state_dict(Node_State_model)

            Node_State_model = copy.deepcopy(self.algo.proj_layer.state_dict())
            Global_State_model = copy.deepcopy(global_node.algo.proj_layer.state_dict())
            for key in Node_State_model.keys():
                Node_State_model[key] = weight * Node_State_model[key] + (1-weight)*Global_State_model[key]
            self.algo.proj_layer.load_state_dict(Node_State_model)
            self.algo.proto.data = weight * self.algo.proto.data + (1-weight)*global_node.algo.proto.data

        else:
  
            self.algo = copy.deepcopy(global_node.algo)
            self.optimizer = init_optimizer(self.algo.model, self.args)

            self.algo.proto.data = copy.deepcopy(global_node.algo.proto.data)





class Global_Node(object):
    def __init__(self, test_all, test_seen, test_novel, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        
        self.algo = Model.build_opencon_model(args)
        self.model = self.algo.model
        
        self.test_all = test_all
        self.test_seen = test_seen
        self.test_novel = test_novel
        self.init = False
        self.save = []
        self.losses = []
    def merge(self, Edge_nodes):
        
        # 融合model
        Node_State_List_model = [copy.deepcopy(Edge_nodes[i].algo.model.state_dict()) for i in range(len(Edge_nodes))]
        self.Dict_model = Node_State_List_model[0]

        for key in self.Dict_model.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict_model[key] += Node_State_List_model[i][key]

            self.Dict_model[key] = self.Dict_model[key].float()     
            self.Dict_model[key] /= len(Edge_nodes)
        self.algo.model.load_state_dict(self.Dict_model)

        Node_State_List_model = [copy.deepcopy(Edge_nodes[i].algo.proj_layer.state_dict()) for i in range(len(Edge_nodes))]
        self.Dict_model = Node_State_List_model[0]

        for key in self.Dict_model.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict_model[key] += Node_State_List_model[i][key]

            self.Dict_model[key] = self.Dict_model[key].float()     
            self.Dict_model[key] /= len(Edge_nodes)
        self.algo.proj_layer.load_state_dict(self.Dict_model)

        self.algo.proto.data = Edge_nodes[0].algo.proto.data
        for i in range(1, len(Edge_nodes)):
                self.algo.proto.data += Edge_nodes[i].algo.proto.data
        self.algo.proto.data = self.algo.proto.data / len(Edge_nodes)
    # def merge(self, Edge_nodes):
        
    #     Node_State_List_model = [copy.deepcopy(Edge_nodes[i].algo.model.state_dict()) for i in range(len(Edge_nodes))]
        
    #     self.Dict_model = Node_State_List_model[0]
        
    #     for key in self.Dict_model.keys():
        
    #         self.Dict_model[key] = self.args.merge_w[0] * self.Dict_model[key]

    #     for key in self.Dict_model.keys():
    #         for i in range(1, len(Edge_nodes)):
    #             self.Dict_model[key] += self.args.merge_w[i]*Node_State_List_model[i][key]

    #         self.Dict_model[key] = self.Dict_model[key].float()     
    #         # self.Dict_model[key] /= len(Edge_nodes)
    #     self.algo.model.load_state_dict(self.Dict_model)

    #     Node_State_List_model = [copy.deepcopy(Edge_nodes[i].algo.proj_layer.state_dict()) for i in range(len(Edge_nodes))]
    #     self.Dict_model = Node_State_List_model[0]

    #     for key in self.Dict_model.keys():
    #         self.Dict_model[key] = self.args.merge_w[0] * self.Dict_model[key]
    #     for key in self.Dict_model.keys():
    #         for i in range(1, len(Edge_nodes)):
    #             self.Dict_model[key] += self.args.merge_w[i]*Node_State_List_model[i][key]

    #         self.Dict_model[key] = self.Dict_model[key].float()     
    #         # self.Dict_model[key] /= len(Edge_nodes)
    #     self.algo.proj_layer.load_state_dict(self.Dict_model)

    #     self.algo.proto.data = self.args.merge_w[0]*Edge_nodes[0].algo.proto.data

    #     for i in range(1, len(Edge_nodes)):
    #             self.algo.proto.data += self.args.merge_w[i]*Edge_nodes[i].algo.proto.data
    #     # self.algo.proto.data = self.algo.proto.data / len(Edge_nodes)
    
    def update(self, Edge_node):

        self.edge_node[Edge_node.num-1] = Edge_node.model

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
    
        

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
