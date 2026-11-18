"""CRIF network: ImageNet ResNet-18 with a small regression head; outputs (standardized log(f/R), alpha /
alpha_scale) for one contact crop, decoded with the run's data.TargetTransform."""
import torch.nn as nn
from torchvision import models


def build_model(pretrained=True, dropout=0.2, hidden=256, output_dim=2):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
    )
    return model


def set_backbone_trainable(model, trainable):
    for name, p in model.named_parameters():
        if not name.startswith('fc.'):
            p.requires_grad = trainable
