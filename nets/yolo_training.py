import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

class YOLOoss(nn.Module):
    def __init__(self,anchors,
                 num_classes,
                 input_shape,
                 cuda,
                 anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]):
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
    
    def MSELoss(self,pred,target):
        return torch.pow(pred - target,2)
    
    def BCELoss(self,pred,target):
        epsilon = 1e-7
        pred =self.clip_by_tensor(pred,epsilon,
                                  1.0 - epsilon)
        output = -target *torch.log(pred) - (1.0 - target) *torch.log(1.0 - pred)
        return output


#计算两个框框的GIoU        
    def box_giou(self,b1,b2):
        """
        input：
        ----------
        b1: tensor, 
        shape=( batch, 
                feat_w, 
                feat_h, 
                anchor_num, 
                4), xywh
        b2: tensor, 
        shape=( batch, 
                feat_w, 
                feat_h, 
                anchor_num, 
                4), xywh

        返回为：
        -------
        giou: tensor, 
        shape=( batch, 
                feat_w, 
                feat_h,
                anchor_num, 
                1)
        """
        
        b1_xy = b1[..., :2]
        b1_wh = b1[... , 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxs = b1_xy +b1_wh_half