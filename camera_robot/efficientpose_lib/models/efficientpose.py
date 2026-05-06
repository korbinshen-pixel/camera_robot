"""
EfficientPose - RGB-D 单流版
输入 4 通道 RGB-D 图像:
- RGB 分支直接用 EfficientNet backbone (in_channels=4)
- 深度作为第 4 个通道在 backbone 里一起提特征
- 多尺度特征 -> BiFPN -> 预测头
"""

import torch
import torch.nn as nn
from .efficientnet import EfficientNetBackbone
from .bifpn import BiFPN


class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class RotationHead(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 6, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class TranslationHead(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class EfficientPose(nn.Module):
    """EfficientPose RGB-D 单流模型"""

    def __init__(self, phi=0, num_classes=1, pretrained=True, in_channels=4):
        """
        Args:
            phi: EfficientNet / BiFPN 尺度
            num_classes: 类别数
            pretrained: 是否加载 ImageNet 预训练权重（只对前 3 通道有意义）
            in_channels: 输入通道数，RGB-D 为 4
        """
        super().__init__()
        self.phi = phi
        self.num_classes = num_classes
        self.in_channels = in_channels

        # EfficientNet backbone，支持自定义 in_channels（4 通道 RGB-D）
        self.backbone = EfficientNetBackbone(
            phi=phi,
            pretrained=pretrained,
            in_channels=in_channels
        )

        # backbone 输出的三个尺度特征通道数 (P3, P4, P5)
        channels_list = self.backbone.feature_info  # e.g. [40, 112, 320] 具体取决于 EfficientNet 实现

        # BiFPN
        self.bifpn_channels = 64 + phi * 16
        self.bifpn = BiFPN(
            in_channels_list=channels_list,
            num_channels=self.bifpn_channels,
            num_layers=3 + phi
        )

        # 预测头
        self.num_anchors = 9
        self.regression_head = RegressionHead(self.bifpn_channels, self.num_anchors)
        self.classification_head = ClassificationHead(self.bifpn_channels, num_classes, self.num_anchors)
        self.rotation_head = RotationHead(self.bifpn_channels, self.num_anchors)
        self.translation_head = TranslationHead(self.bifpn_channels, self.num_anchors)

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W]  输入 RGB-D (前三通道为 RGB，第四通道为深度归一化)
        Returns:
            dict:
              'bbox': [B, N, 4]
              'class': [B, N, num_classes]
              'rotation': [B, N, 6]
              'translation': [B, N, 3]
        """
        # backbone 多尺度特征 [P3, P4, P5]
        feats = self.backbone(x)  # list of 3 feature maps

        # BiFPN 融合
        features = self.bifpn(feats)
        feat = features[0]  # 取最高分辨率 F3 作为预测头输入（按你原来的写法）

        # 预测头
        bbox_pred = self.regression_head(feat)
        class_pred = self.classification_head(feat)
        rotation_pred = self.rotation_head(feat)
        translation_pred = self.translation_head(feat)

        B = x.shape[0]
        H, W = feat.shape[-2:]
        N = self.num_anchors * H * W

        # 维度整理 [B, C, H, W] -> [B, N, *]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, N, 4)
        class_pred = class_pred.permute(0, 2, 3, 1).reshape(B, N, self.num_classes)
        rotation_pred = rotation_pred.permute(0, 2, 3, 1).reshape(B, N, 6)
        translation_pred = translation_pred.permute(0, 2, 3, 1).reshape(B, N, 3)

        return {
            'bbox': bbox_pred,
            'class': class_pred,
            'rotation': rotation_pred,
            'translation': translation_pred
        }

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    # 简单自测
    model = EfficientPose(phi=0, num_classes=1, pretrained=False, in_channels=4)
    x = torch.randn(2, 4, 512, 512)
    outputs = model(x)

    print("EfficientPose RGB-D 单流测试:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")