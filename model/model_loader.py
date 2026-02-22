import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 17

def load_model(path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model