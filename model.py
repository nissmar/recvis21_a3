import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20


class Net0(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def Net_101():
    model = models.resnet101(pretrained=True)
    n_layer = 0
    for param in model.parameters():
        n_layer += 1
        if n_layer < 314 - 11:
            param.requires_grad = False
    model.fc = nn.Linear(2048, nclasses, bias=True)

    return model


def Net():
    model = models.resnet50(pretrained=True)
    n_layer = 0
    for param in model.parameters():
        n_layer += 1
        param.requires_grad = False
        # else:
        #     print(param.shape)
    model.fc = nn.Sequential(
        nn.Linear(2048, 100, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(100, 100, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(100, 100, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(100, nclasses, bias=True),
    )
    return model

