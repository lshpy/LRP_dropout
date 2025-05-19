import torchvision.models as models
import torch.nn as nn

class ResNetWithHook(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 마지막 conv
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = resnet.fc

    def extract_features(self, x):
        return self.feature_extractor(x)  # shape: (B, C, H, W)

    def classify_from_features(self, features):
        pooled = self.pool(features).view(features.size(0), -1)
        return self.fc(pooled)

    def forward(self, x):
        return self.classify_from_features(self.extract_features(x))
