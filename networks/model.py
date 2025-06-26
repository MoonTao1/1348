import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.backbone import build_backbone
from functions import ReverseLayerF
from torchvision.transforms import Resize
from clip import *
# clip_path = 'clip'
# device = torch.device('cuda:0')
# backbone = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
# id = 4
class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256 * 16 * 16, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.DSCClassifier = DSCClassifier(in_channels=256, num_classes=2)

        self.crossModelAtt =CrossModelAtt()
        self.conv = nn.Conv2d(512, 512 // 2, kernel_size=1, stride=1, padding=0)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input,alpha):

        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x_clip = self.crossModelAtt(input, x)
        x = torch.cat([x, x_clip], dim=1)

        # print(x1.shape)
        x = self.conv(x)
        # feature = x.view(x.size(0), -1)
        feature = x
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.DSCClassifier(reverse_feature)

        x, features = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, domain_output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class CrossModelAtt(nn.Module):
    def __init__(self, backbone="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(backbone, device=device)
        self.model = self.model.eval()

        self.proj = nn.Linear(512, 256)  # CLIP 512D → 256D
        self.conv = nn.Conv2d(256, 256, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.resize = Resize([224, 224])

    def forward(self, img, feature):
        b, c, h, w = feature.shape  # feature: [16, 256, 16, 16]

        # 1: 处理 CLIP 特征
        img = self.resize(img)  # 调整大小到 CLIP 输入尺寸
        with torch.no_grad():
            clip_feature = self.model.encode_image(img) .float() # [B, 512]

        clip_feature = self.proj(clip_feature)  # 降维到 256D: [B, 256]
        clip_feature = clip_feature.view(b, c, 1, 1)  # [B, 256, 1, 1]
        clip_feature = clip_feature.expand(b, c, h, w)  # [B, 256, 16, 16]

        # 2: 计算注意力权重
        q = clip_feature.view(b, c, -1)  # [B, 256, 256]
        k = clip_feature.view(b, c, -1).permute(0, 2, 1)  # [B, 256, 256]

        perception = torch.bmm(q, k)  # [B, 256, 256]
        perception = torch.max(perception, -1, keepdim=True)[0].expand_as(perception) - perception

        v = feature.view(b, c, -1)  # [B, 256, 256]
        perception_info = torch.bmm(perception.float(), v.float())  # [B, 256, 256]
        perception_info = perception_info.view(b, c, h, w)  # [B, 256, 16, 16]

        # 3: 加入输入特征
        perception_info = self.gamma * perception_info + feature  # 残差连接

        return perception_info
class DSCClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(DSCClassifier, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 深度卷积
        self.pointwise = nn.Conv2d(in_channels, 128, kernel_size=1)  # 逐点卷积降维
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.LogSoftmax(dim=1)
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
    def forward(self, x):
        x = self.depthwise(x)  # 先做深度卷积，提取局部特征
        x = self.pointwise(x)  # 再做逐点卷积，调整通道数
        x = self.gap(x)  # 全局平均池化
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.domain_classifier(x)

        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == '__main__':
    # model = DeepLab(num_classes=1, backbone='mobilenet', output_stride=16,
    #                 sync_bn=True, freeze_bn=False)
    model, _ = clip.load("ViT-B/32", "cuda")
    count_parameters(model)

# class DSCClassifier(nn.Module):
#     def __init__(self, in_channels, num_classes=2):
#         super(DSCClassifier, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)  # 深度卷积
#         self.pointwise = nn.Conv2d(in_channels, 128, kernel_size=1)  # 逐点卷积降维
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = self.depthwise(x)  # 先做深度卷积，提取局部特征
#         x = self.pointwise(x)  # 再做逐点卷积，调整通道数
#         x = self.gap(x)  # 全局平均池化
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)


