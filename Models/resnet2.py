import torch.nn as nn
import torch.nn.functional as F

from Models.model import WeightNormLayer

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # self.nin = nn.Conv2d(planes, planes, 1)
        # self.activation = topkrelu()
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        # out = self.activation(self.nin(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.activation(out)
        # out = self.activation(self.nin(out))
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size, head_name='linear'):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size
        self.head_name = head_name

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion
        last_hid = last_hid * (self.input_size[-1] // 2 // 2 // 2 // 4) ** 2
        self.features_size = last_hid

        if self.head_name == "weightnorm":
            self.head = WeightNormLayer(self.features_size, num_classes, bias=False)
        else:
            self.head = nn.Linear(last_hid, num_classes)
        self.activation = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def feature_extractor(self, x):
        bsz = x.size(0)
        assert x.ndim == 4
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.adaptive_avg_pool2d(out, 1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.head(out)
        return out

def ResNet18_2(nclasses, head_name, nf=20, input_size=(3, 32, 32), *args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size, head_name, *args, **kwargs)
