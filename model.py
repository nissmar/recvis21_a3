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

def Net_101():
    model = models.resnet101(pretrained=True)
    requires_grad = ['layer4.2.conv1.weight', 'layer4.2.conv2.weight','layer4.2.conv3.weight']
  
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

    for name, param in model.named_parameters():
        if not(name in requires_grad):
            param.requires_grad = False

    model.fc = nn.Linear(2048, nclasses, bias=True)
    return model


# model = load_res_50()
# print([p.shape for p in model.parameters() if p.requires_grad])
# print(number_of_params([p.shape for p in model.parameters() if p.requires_grad]))

# print(model(torch.rand((1,3,224,224))))