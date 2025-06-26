import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class SalMAEodel(nn.Module):
    def __init__(self):
        n, m = 24, 3

        super(SalMAEodel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)

        # encoder ===============================
        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd4 = conv3x3(4*n, 4*n)

        # decoder ===============================
        self.convu3 = conv3x3(8*n, 4*n)
        self.convu2 = conv3x3(6*n, 2*n)
        self.convu1 = conv3x3(3*n, 1*n)

        self.convu0_sal = nn.Conv2d(n, 1, 3, 1, 1)

        # decoder ===============================
        self.convu3_ = conv3x3(4*n, 4*n)
        self.convu2_ = conv3x3(4*n, 2*n)
        self.convu1_ = conv3x3(2*n, 1*n)
        self.convu0_con = nn.Conv2d(n, 3, 3, 1, 1)

    def encoder(self, x):
        x1 = self.convd1(x)

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)
        return x1, x2, x3, x4

    def sal_decoder(self, x):
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)

        y1 = self.convu0_sal(y1)
        y1 = self.sigmoid(y1)
        return y1

    def con_decoder(self, x):
        z3 = self.upsample(x)
        z3 = self.convu3_(z3)

        z2 = self.upsample(z3)
        z2 = self.convu2_(z2)

        z1 = self.upsample(z2)
        z1 = self.convu1_(z1)

        z1 = self.convu0_con(z1)
        return z1

    def forward(self, x):
        x_encoder = self.encoder(x)
        img_sal = self.sal_decoder(x_encoder)

        img_mask = (1 - img_sal) * x

        y_encoder = self.encoder(img_mask)
        y_encoder = y_encoder[-1]
        img_con = self.con_decoder(y_encoder)

        return img_sal, img_con


if __name__ == '__main__':
    b, c, h, w = 1, 3, 224, 224
    img = torch.rand(size=(b, c, h, w))

    model = Model()
    img_sal, img_con = model(img)
    print(img_sal.shape, img_con.shape)

    '''
    # Para====================================
    '''
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count/10e6} M")
    import time
    import torch

    '''
    # FPS====================================
    '''
    import torch
    import time
    model.eval()  # 切换到评估模式

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_data = img.to(device)

    # 推理前的 warm-up 避免初次运行时间偏长
    with torch.no_grad():
        for _ in range(10):  # 模拟10次推理作为预热
            _ = model(input_data)

    # 正式推理时间测量
    num_iterations = 1000   # 设置测量多少次推理
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)

    end_time = time.time()
    total_time = end_time - start_time

    total_frames = num_iterations * input_data.size(0)
    fps = total_frames / total_time

    print(f"FPS: {fps:.2f}")

    '''
    # FLOPs====================================
    '''
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model.cpu(), img.cpu())
    print(f"FLOPs: {flops.total()/10e9} G")  # 输出总的FLOPs
