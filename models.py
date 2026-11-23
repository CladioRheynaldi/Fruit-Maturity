import torch
from torch import nn
from torchvision import models, transforms

CLASSES = ['unripe', 'ripe', 'overripe', 'rotten']
SIZE = 128

def transform(augment=False):
    ops = [transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC)]
    if augment:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(12)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        layers = []
        channels = 3
        for width in [16, 32, 64]:
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.head(self.features(x))

def build(architecture, pretrained=False, num_classes=4):
    if architecture == 'cnn':
        return SmallCNN(num_classes)
    if architecture == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f'Unknown architecture: {architecture}')

def head_and_features(model, architecture):
    if architecture == 'resnet18':
        return model.fc, nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return model.classifier, nn.Sequential(model.features, model.avgpool, nn.Flatten())
