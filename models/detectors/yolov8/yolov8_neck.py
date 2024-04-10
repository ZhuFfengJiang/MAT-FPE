import torch
import torch.nn as nn
from .FPE import EVCBlock

try:
    from .yolov8_basic import Conv
except:
    from yolov8_basic import Conv


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """

    def __init__(self, cfg, in_dim, out_dim, expand_ratio=0.5):
        super().__init__()
        
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.m = nn.MaxPool2d(kernel_size=cfg['pooling_size'], stride=1, padding=cfg['pooling_size'] // 2)
        self.EVC = EVCBlock(in_channels=256, out_channels=256)
        #self.qf = SwinTransformerBlock(dim=256,num_heads=6,window_size=15,input_resolution=((60 //4),(60 // 4)))
    
    def forward(self, x):
        
        x = self.cv1(x)  #torch.Size([1, 128, 30, 30])
        y1 = self.m(x)   #torch.Size([1, 128, 30, 30])
        y2 = self.m(y1)  #torch.Size([1, 128, 30, 30])
        z = self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))#torch.Size([1, 256, 30, 30])
        e = self.EVC(z)
        #torch.Size([1, 256, 30, 30])

        return e
        """X: torch.Size([1, 128, 30, 30])
        y1: torch.Size([1, 128, 30, 30])
        y2: torch.Size([1, 128, 30, 30])
        z: torch.Size([1, 256, 30, 30])
        zevc: torch.Size([1, 256, 30, 30])"""
        # return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        # （1，256，60，60）


# SPPF block with CSP module
class SPPFBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """

    def __init__(self, cfg, in_dim, out_dim, expand_ratio):
        super(SPPFBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])
        self.m = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1,
                 act_type=cfg['neck_act'], norm_type=cfg['neck_norm'],
                 depthwise=cfg['neck_depthwise']),
            SPPF(cfg, inter_dim, inter_dim, expand_ratio=1.0),
            Conv(inter_dim, inter_dim, k=3, p=1,
                 act_type=cfg['neck_act'], norm_type=cfg['neck_norm'],
                 depthwise=cfg['neck_depthwise'])
        )
        self.cv3 = Conv(inter_dim * 2, self.out_dim, k=1, act_type=cfg['neck_act'], norm_type=cfg['neck_norm'])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'sppf':
        neck = SPPF(cfg, in_dim, out_dim, cfg['neck_expand_ratio'])
    elif model == 'csp_sppf':
        neck = SPPFBlockCSP(cfg, in_dim, out_dim, cfg['neck_expand_ratio'])
    return neck

