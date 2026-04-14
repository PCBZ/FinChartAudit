"""Shared ViT model definition — used by both training and API inference."""

import torch.nn as nn
import timm

from src.prompts import MISLEADER_TYPES

TYPE_TO_IDX = {t: i for i, t in enumerate(MISLEADER_TYPES)}
NUM_CLASSES = len(MISLEADER_TYPES)


def build_vit_model(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    """Build ViT-B/16 with classification head matching checkpoint format."""
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
    model.head = nn.Sequential(
        nn.Linear(model.num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model
