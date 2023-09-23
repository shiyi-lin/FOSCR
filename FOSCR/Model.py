import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from easyfl.models.model import BaseModel
from easyfl.models.resnet import ResNet18, ResNet50
import torch.distributed as dist


OneLayer = "1_layer"
TwoLayer = "2_layer"


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def build_byol_model(args):
    
    if args.arch == 'resnet18':
        net = ResNet18(num_classes=args.all_class, inplane=args.input_planes)
    elif args.arch == 'resnet50':
        net = ResNet50(num_classes=args.all_class)

    net = net.cuda()

    # # use dataparallel if there's multiple gpus
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    #     simnet = torch.nn.DataParallel(simnet)
    return BYOLModel(net=net, stop_gradient=True, has_predictor=True,
                         predictor_network=TwoLayer, all_class=args.all_class)

def build_opencon_model(args):
    from models.OpenSupCon import OpenSupCon
    algo = OpenSupCon("RN18_simclr_CIFAR", args)
    
    return algo


# ------------- BYOL Model -----------------
# class BaseModel(nn.Module):
#     def __init__(self):
#         super(BaseModel, self).__init__()

class BYOLModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
            all_class=10
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP_byol(feature_dim, projection_size, projection_hidden_size).cuda()  # projector
        self.fc = nn.Linear(projection_size, all_class, bias=False).cuda()

        self.online_predictor = MLP_byol(projection_size, projection_size, projection_hidden_size, predictor_network).cuda()
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # debug purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two):
        online_pred_one = self.online_encoder(image_one)
        online_pred_two = self.online_encoder(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        if self.stop_gradient:
            with torch.no_grad():
                if self.target_encoder is None:
                    self.target_encoder = self._get_target_encoder()
                target_proj_one = self.target_encoder(image_one)
                target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()

        else:
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            target_proj_one = self.target_encoder(image_one)
            target_proj_two = self.target_encoder(image_two)
        online_linear_one = self.fc(online_pred_one)
        online_linear_two = self.fc(online_pred_two)

        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two

        return online_linear_one, online_linear_two, loss.mean()


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class MLP_byol(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)
