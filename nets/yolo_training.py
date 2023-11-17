import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

class YOLOoss(nn.Module):
    def __init__(self,anchors, num_classes,input_shape,cuda,anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]):
        super(YOLOoss,self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attars = num_classes+5
        self.input_shape = input_shape
        self.anchor_mask = anchor_mask
        
        self.giou =True
        self.balance =[0.4,1.0,4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)
        
        self.ignore_threshold = 0.5
        self.cuda = cuda
    def clip_by_tensor(self,t,t_min,t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t< t_min).float() * t_min
        result = (result <= t_max).float() *result +(result >t_max ).float()*t_max
        return result