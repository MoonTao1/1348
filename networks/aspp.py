import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from clip import *
from torchvision.transforms import Resize

clip_path = 'clip'
device = torch.device('cuda:0')
backbone = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
id = 4

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)

        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()
        # self.crossModelAtt = CrossModelAtt()
    def forward(self,x):
        x1 = self.aspp1(x)

        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



# class CrossModelAtt(nn.Module):
#     def __init__(self, backbone="ViT-B/32", device="cuda"):
#         super().__init__()
#         self.device = device
#         self.model, _ = clip.load(backbone, device=device)
#         self.model = self.model.eval()
#
#         self.proj = nn.Linear(512, 256)  # CLIP 512D → 256D
#         self.conv = nn.Conv2d(256, 256, kernel_size=1)
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
#         self.resize = Resize([224, 224])
#
#     def forward(self, img, feature):
#         b, c, h, w = feature.shape  # feature: [16, 256, 16, 16]
#
#         # 1: 处理 CLIP 特征
#         img = self.resize(img)  # 调整大小到 CLIP 输入尺寸
#         with torch.no_grad():
#             clip_feature = self.model.encode_image(img) .float() # [B, 512]
#
#         clip_feature = self.proj(clip_feature)  # 降维到 256D: [B, 256]
#         clip_feature = clip_feature.view(b, c, 1, 1)  # [B, 256, 1, 1]
#         clip_feature = clip_feature.expand(b, c, h, w)  # [B, 256, 16, 16]
#
#         # 2: 计算注意力权重
#         q = clip_feature.view(b, c, -1)  # [B, 256, 256]
#         k = clip_feature.view(b, c, -1).permute(0, 2, 1)  # [B, 256, 256]
#
#         perception = torch.bmm(q, k)  # [B, 256, 256]
#         perception = torch.max(perception, -1, keepdim=True)[0].expand_as(perception) - perception
#
#         v = feature.view(b, c, -1)  # [B, 256, 256]
#         perception_info = torch.bmm(perception.float(), v.float())  # [B, 256, 256]
#         perception_info = perception_info.view(b, c, h, w)  # [B, 256, 16, 16]
#
#         # 3: 加入输入特征
#         perception_info = self.gamma * perception_info + feature  # 残差连接
#
#         return perception_info


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)
