import torch
import torch.nn as nn
from .MTA import DilateBlock
#try:
from .yolov8_basic import  Yolov8StageBlock
#except:
#    from yolov8_basic import  Yolov8StageBlock

try:
    from .yolov8_basic import ConvDCN as Conv
except:
    from yolov8_basic import ConvDCN as Conv
import torch.nn as nn

# ---------------------------- Basic functions ----------------------------

class Yolov8Backbone(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(Yolov8Backbone, self).__init__()
        self.feat_dims = [round(64 * width), round(128 * width), round(256 * width), round(512 * width),
                          round(512 * width * ratio)]
        # P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        # P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim=self.feat_dims[1],
                             out_dim=self.feat_dims[1],
                             num_blocks=round(3 * depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim=self.feat_dims[2],
                             out_dim=self.feat_dims[2],
                             num_blocks=round(6 * depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim=self.feat_dims[3],
                             out_dim=self.feat_dims[3],
                             num_blocks=round(6 * depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolov8StageBlock(in_dim=self.feat_dims[4],
                             out_dim=self.feat_dims[4],
                             num_blocks=round(3 * depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        self.m1 = DilateBlock(dim=64,num_heads=8,dilation=[1,2,3,4],cpe_per_block=True)
        self.m2 = DilateBlock(dim=128,num_heads=8,dilation=[1,2,3,4],cpe_per_block=True)
        self.m3 = DilateBlock(dim=256,num_heads=8,dilation=[1,2,3,4],cpe_per_block=True)
    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c33 = self.layer_3(c2)
        c3 = self.m1(c33)
        c44 = self.layer_4(c3)
        c4 = self.m2(c44)
        c55 = self.layer_5(c4)
        c5 = self.m3(c55)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## build Yolov8's Backbone
def build_backbone(cfg):
    # model
    
    backbone = Yolov8Backbone(width=cfg['width'],
                              depth=cfg['depth'],
                              ratio=cfg['ratio'],
                              act_type=cfg['bk_act'],
                              norm_type=cfg['bk_norm'],
                              depthwise=cfg['bk_depthwise']
                              )
    feat_dims = backbone.feat_dims[-3:]

    return backbone, feat_dims


