"""Model factories for occlusion-fer."""

from torch import nn
from torchvision import models as tv_models


def create_resnet18(
    num_classes: int = 7,
    pretrained: bool = True,
) -> nn.Module:
    """Create a standard ResNet-18 that returns raw class logits."""
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if type(pretrained) is not bool:
        raise ValueError("pretrained must be a bool")

    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    if pretrained:
        try:
            model = tv_models.resnet18(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained ResNet-18 weights; "
                "random initialization was not used"
            ) from exc
    else:
        model = tv_models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
