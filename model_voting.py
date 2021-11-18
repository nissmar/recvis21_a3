import torch
from model import Net


model = Net()
model.load_state_dict(
    torch.load("models/res101p86.pth", map_location=torch.device("cpu"))
)
model.eval()

model2 = Net()
model2.load_state_dict(
    torch.load("models/res101p86_2.pth", map_location=torch.device("cpu"))
)
model2.eval()

model3 = Net()
model3.load_state_dict(
    torch.load("models/res101p86_3.pth", map_location=torch.device("cpu"))
)
model3.eval()


models = [model, model2, model3]


def ensemble_voting(data):
    outputs = []
    for model in models:
        outputs.append(model(data).data.max(1)[1])
    outputs.sort()
    if outputs[0] == outputs[1]:
        return outputs[0]
    return outputs[2]

