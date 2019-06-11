from torchvision.models.resnet import ResNet
from torch import nn


def update_number_of_classes(model: ResNet, new_number_of_classes) -> ResNet:
    model.fc = nn.Linear(model.fc.in_features, new_number_of_classes)
    return model
