import zipfile
import os

import torchvision.transforms as transforms


my_mean = [0.5079, 0.5221, 0.4361]
my_std = [0.2079, 0.2069, 0.2217]

data_transforms_train = transforms.Compose(
    [
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=my_mean, std=my_std),


    ]
)
data_transforms = transforms.Compose(
    [
      
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=my_mean, std=my_std),
    ]
)