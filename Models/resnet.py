"""
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
In CVPR, 2016.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from Models.model import WeightNormLayer

__all__ = [
    "CifarResNet",
    "cifar_resnet20",
    "cifar_resnet32",
    "cifar_resnet44",
    "cifar_resnet56",
]

pretrained_settings = {
    "CIFAR10": {
        "resnet20": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt",
        "resnet32": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt",
        "resnet44": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt",
        "resnet56": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt",
        "num_classes": 10,
    },
    "CIFAR100": {
        "resnet20": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt",
        "resnet32": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt",
        "resnet44": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt",
        "resnet56": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt",
        "num_classes": 100,
    },
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class CifarResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        dropout=False,
        head_name="linear",
    ):
        super(CifarResNet, self).__init__()
        self.head_name = head_name
        self.image_size = 32
        self.input_dim = 3
        self.data_shape = [self.input_dim, self.image_size, self.image_size]
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        self.data_encoded = False
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.features_size = 64 * block.expansion
        self.classes_mask = torch.eye(self.num_classes).cuda().float()

        if self.head_name == "weightnorm":
            self.head = WeightNormLayer(self.features_size, num_classes, bias=False)
        else:
            self.head = nn.Linear(self.features_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_last_layer(self):
        return self.head.layer

    def set_data_encoded(self, flag):
        self.data_encoded = flag

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_task(self, x, task_ids):

        if not self.data_encoded:
            x = x.view(-1, self.input_dim, self.image_size, self.image_size)
            x = self.feature_extractor(x)
        if self.dropout is not None:
            x = self.dropout_layer(x)
        x = self.head.forward_task(x, task_ids)
        return x

    def encoder(self, x):
        return self.feature_extractor(x)

    def forward(self, x):

        if not self.data_encoded:
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        if self.dropout is not None:
            x = self.dropout_layer(x)
        assert x.shape[-1] == self.num_classes, print(
            f"{x.shape[-1]} vs {self.num_classes}"
        )

        return x

    def accumulate(self, batch, labels, epoch=0):

        if not self.data_encoded:
            batch = self.feature_extractor(batch)

        batch = batch.view(batch.size(0), -1)
        self.get_last_layer().accumulate(batch, labels, epoch)

    def update_head(self, epoch=0):
        self.get_last_layer().update(epoch)


def cifar_resnet20(
    num_classes=10,
    pretrained=None,
    model_dir=None,
    dropout=None,
    head_name="linear",
    **kwargs,
):
    if pretrained is None:
        model = CifarResNet(
            BasicBlock,
            [3, 3, 3],
            num_classes=num_classes,
            dropout=dropout,
            head_name=head_name,
            **kwargs,
        )
    else:
        model = CifarResNet(
            BasicBlock,
            [3, 3, 3],
            num_classes=pretrained_settings[pretrained]["num_classes"],
            dropout=dropout,
            head_name=head_name,
        )
        model.load_state_dict(
            model_zoo.load_url(
                pretrained_settings[pretrained]["resnet20"], model_dir=model_dir
            )
        )
    return model


def cifar_resnet32(
    num_classes=10,
    pretrained=None,
    model_dir=None,
    dropout=None,
    head_name="linear",
    **kwargs,
):
    if pretrained is None:
        model = CifarResNet(
            BasicBlock,
            [5, 5, 5],
            num_classes=num_classes,
            dropout=dropout,
            head_name=head_name,
            **kwargs,
        )
    else:
        model = CifarResNet(
            BasicBlock,
            [5, 5, 5],
            num_classes=pretrained_settings[pretrained]["num_classes"],
        )
        model.load_state_dict(
            model_zoo.load_url(
                pretrained_settings[pretrained]["resnet32"], model_dir=model_dir
            )
        )
    return model


def cifar_resnet44(
    num_classes=10,
    pretrained=None,
    model_dir=None,
    dropout=None,
    head_name="linear",
    masking="None",
    **kwargs,
):
    if pretrained is None:
        model = CifarResNet(
            BasicBlock,
            [7, 7, 7],
            num_classes=num_classes,
            dropout=dropout,
            head_name=head_name,
            masking=masking,
            **kwargs,
        )
    else:
        model = CifarResNet(
            BasicBlock,
            [7, 7, 7],
            num_classes=pretrained_settings[pretrained]["num_classes"],
        )
        model.load_state_dict(
            model_zoo.load_url(
                pretrained_settings[pretrained]["resnet44"], model_dir=model_dir
            )
        )
    return model


def cifar_resnet56(
    num_classes=10,
    pretrained=None,
    model_dir=None,
    dropout=None,
    head_name="linear",
    masking="None",
    **kwargs,
):
    if pretrained is None:
        model = CifarResNet(
            BasicBlock,
            [9, 9, 9],
            num_classes=num_classes,
            dropout=dropout,
            head_name=head_name,
            masking=masking,
            **kwargs,
        )
    else:
        model = CifarResNet(
            BasicBlock,
            [9, 9, 9],
            num_classes=pretrained_settings[pretrained]["num_classes"],
        )
        model.load_state_dict(
            model_zoo.load_url(
                pretrained_settings[pretrained]["resnet56"], model_dir=model_dir
            )
        )
    return model