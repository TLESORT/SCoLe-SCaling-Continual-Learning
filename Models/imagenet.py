import torch.nn as nn
from torchvision import models
import torch

from Models.model import WeightNormLayer

class ImageNetModel(nn.Module):

    def __init__(self, num_classes=10,
                 pretrained=False,
                 name_model="resnet",
                 head_name="linear"):
        super(ImageNetModel, self).__init__()
        self.pretrained = pretrained
        self.name_model = name_model
        self.num_classes = num_classes
        self.data_encoded = False
        self.image_size = 224
        self.input_dim = 3
        self.data_shape = [self.input_dim, self.image_size, self.image_size]
        self.head_name = head_name

        if self.name_model == "alexnet":
            model = models.alexnet(pretrained=True)
            self.latent_dim = list(model.children())[-1][-1].in_features #2048
            self.classifier = nn.Sequential(*list(model.children())[-1][:-1])  # between features and outlayer
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.features_size = 9216
        elif self.name_model == "resnet":
            model = models.resnet18(pretrained=True)
            self.latent_dim = list(model.children())[-1].in_features  #512
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.features_size = 512
        elif self.name_model == "googlenet":
            model = models.googlenet(pretrained=True)
            self.latent_dim = list(model.children())[-1].in_features # 1024
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.features_size = 1024
        elif self.name_model == "vgg":
            model = models.vgg16(pretrained=True)
            self.latent_dim = list(model.children())[-1][-1].in_features #2048
            self.classifier = nn.Sequential(*list(model.children())[-1][:-1])
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.features_size = 25088
        else:
            raise Exception("Ca va pas la")

        if self.head_name == "weightnorm":
            self.head = WeightNormLayer(self.features_size, num_classes, bias=False)
        else:
            self.head = nn.Linear(self.features_size, num_classes)


    def set_data_encoded(self, flag):
        self.data_encoded = flag
        if self.data_encoded:
            # we can free some memory if the data is already encoded
            self.classifier = None
            self.features = None

    def get_last_layer(self):
        return self.head.layer

    def feature_extractor(self, x):

        if self.name_model in ["alexnet", "vgg"]:
            x = self.classifier(self.features(x).view(-1, self.features_size))
        else:
            x = self.features(x)

        x = x.view(-1, self.latent_dim)
        return x

    def forward_task(self, x, task_ids):

        if not self.data_encoded:
            x = x.view(-1, self.input_dim, self.image_size, self.image_size)
            x = self.feature_extractor(x)
        x = self.head.forward_task(x, task_ids)
        return x

    def forward(self, x):
        if not self.data_encoded:
            x = x.view(-1, self.input_dim, self.image_size, self.image_size)
            x = self.feature_extractor(x)
        x = x.view(-1, self.latent_dim)
        return self.head(x)

    def accumulate(self, batch, labels, epoch=0):

        if not self.data_encoded:
            batch = self.feature_extractor(batch)

        batch = batch.view(batch.size(0), -1)
        self.get_last_layer().accumulate(batch, labels, epoch)

    def update_head(self, epoch):
        self.get_last_layer().update(epoch)

    def get_loss(self, out, labels, loss_func):
        if self.masking == "single":
            out = torch.mul(out, self.classes_mask[labels])
        elif self.masking == "group":
            label_unique = labels.unique()
            ind_mask = self.classes_mask[label_unique].sum(0)
            full_mask = ind_mask.unsqueeze(0).repeat(out.shape[0], 1)
            out = torch.mul(out, full_mask)
        loss = loss_func(out, labels.long())
        assert loss == loss, print("There should be some Nan")
        return loss
