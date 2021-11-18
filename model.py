import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20

def number_of_params(arr):
    s=0
    for e in arr:
        c=1
        for x in e:
            c*=x
        s+=c
    return s


def conv_to_activate(arr):
    '''arr: [(layer,sublayer,name)]'''
    return ["layer"+str(e[0])+"."+str(e[1])+".conv"+str(e[2])+".weight" for e in arr]

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

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        
def Net_vgg():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    headModel = nn.Sequential(
        nn.Linear(25088, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(2048, 20)
    )
    model.classifier = headModel
    return model

def Net_res101():
    model = models.resnet101(pretrained=True)
    n_layer = 0
    for param in model.parameters():
        n_layer += 1
        if n_layer < 314 - 11:
            param.requires_grad = False
    model.fc = nn.Linear(2048, nclasses, bias=True)
    model.apply(set_bn_eval)

    return model

def Net_incep():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.AuxLogits.fc = nn.Linear(768, nclasses)
    model.fc = nn.Linear(2048, nclasses)

    return model



def loadNet(str, train_lay):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(2048, nclasses, bias=True)
    model.load_state_dict(torch.load(str))

    requires_grad = ['layer4.1.conv2.weight', 'layer4.1.conv3.weight', 'layer4.2.conv1.weight', 'layer4.2.conv2.weight','layer4.2.conv3.weight','fc.weight','fc.bias'][5-train_lay:]
    for name, param in model.named_parameters():
        if not(name in requires_grad):
            param.requires_grad = False
        
    print('MODEL LOADED',str)
    print([p.shape for p in model.parameters() if p.requires_grad])

    return model

def Net_1012():
    model = models.resnet101(pretrained=True)
    n_layer = 0
    requires_grad = ['layer4.2.conv2.weight','layer4.2.conv3.weight']
    # requires_grad = ['layer1.0.conv1.weight']
    # requires_grad = []

    for name, param in model.named_parameters():
        if not(name in requires_grad):
            param.requires_grad = False
    model.fc = nn.Linear(2048, nclasses, bias=True)
    return model

def load_res_50(str = "models/res50_91.pth"):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, nclasses, bias=True)
    model.load_state_dict(torch.load(str))
    return model
    
def Net():
    model = models.resnet50(pretrained=True)
    requires_grad = ['layer4.1.conv1.weight','layer4.1.conv2.weight','layer4.1.conv3.weight']
    # requires_grad = ['layer1.0.conv1.weight']
    # requires_grad = []

    for name, param in model.named_parameters():
        if not(name in requires_grad):
            param.requires_grad = False

    model.fc = nn.Linear(2048, nclasses, bias=True)
    return model


# model = load_res_50()
# print([p.shape for p in model.parameters() if p.requires_grad])
# print(number_of_params([p.shape for p in model.parameters() if p.requires_grad]))

# print(model(torch.rand((1,3,224,224))))