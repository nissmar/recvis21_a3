import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
from torchvision import datasets
from data import data_transforms
import torch
from model import Net101_18, Net101_2



val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder("bird_dataset/val_images", transform=data_transforms),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

use_cuda = torch.cuda.is_available()

state_dict = torch.load("models/res101_18_97.pth")
model18 = Net101_18()
model18.load_state_dict(state_dict)
model18.cuda()
model18.eval()

state_dict = torch.load("models/res101crows85.pth")
modelcrows = Net101_2()
modelcrows.load_state_dict(state_dict)
modelcrows.cuda()
modelcrows.eval()



state_dict = torch.load("models/res101cuckoo100.pth")
modelcuckoo = Net101_2()
modelcuckoo.load_state_dict(state_dict)
modelcuckoo.cuda()
modelcuckoo.eval()

print('ok')
def find_output(data):
    output = model18(data)
    pred = int(output.data.max(1, keepdim=True)[1])
    if pred<15:
        return pred
    elif pred==15:
        #crow
        output2 = modelcrows(data)
        pred2 = int(output2.data.max(1, keepdim=True)[1])
        return pred + pred2
    elif pred==16:
        #cuckoo
        output2 = modelcuckoo(data)
        pred2 = int(output2.data.max(1, keepdim=True)[1])
        return pred+1 + pred2
    else:
        return pred+2


def validation():
    correct = 0

    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        pred = find_output(data)
        print(pred,int(target))
        correct += (pred == int(target))
    return (100.0 * correct / len(val_loader.dataset))


test_dir = "bird_dataset/test_images/mistery_category"


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


output_file = open("experiment/kaggle.csv", "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if "jpg" in f:
        data = data_transforms(pil_loader(test_dir + "/" + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        # output = model(data)
        # pred = output.data.max(1, keepdim=True)[1]
        pred = find_output(data)
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print(
    "Succesfully wrote "
    + args.outfile
    + ", you can upload this file to the kaggle competition website"
)

# print(validation())