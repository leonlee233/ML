import torch
import torch.nn as nn
import numpy as np
from utils.utils import bboxes_iou


class YOLOLayer(nn.Module):
    """
    对应于darknet的yolo_layer.c的检测层
    """
    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        YOLOLayer类的初始化。
        Args:
            config_model (dict) : 模型配置。
                ANCHORS (元组列表) :
                ANCH_MASK (整数列表的列表): 指示在YOLO层中要使用的锚点的索引。从列表中选择一个掩码组。
                N_CLASSES (int): 类别数
            layer_no (int): YOLO层号 - 从(0, 1, 2)中选择一个。
            in_ch (int): 输入通道数。
            ignore_thre (float): IoU的阈值，高于此阈值的对象性训练将被忽略。
        """

        super(YOLOLayer, self).__init__()  # 调用父类nn.Module的初始化方法

        if config_model['TYPE'] == 'YOLOv3':
            self.module_list = create_yolov3_modules(config_model, ignore_thre)  # 创建YOLOv3模块
        else:
            raise Exception('Model name {} is not available'.format(config_model['TYPE']))  # 如果模型类型不是YOLOv3，抛出异常

        strides = [32, 16, 8]  # 固定步长
        self.anchors = config_model['ANCHORS']  # 锚点
        self.anch_mask = config_model['ANCH_MASK'][layer_no]  # 锚点掩码
        self.n_anchors = len(self.anch_mask)  # 锚点数量
        self.n_classes = config_model['N_CLASSES']  # 类别数量
        self.ignore_thre = ignore_thre  # 忽略阈值
        self.l2_loss = nn.MSELoss(size_average=False)  # L2损失
        self.bce_loss = nn.BCELoss(size_average=False)  # 二元交叉熵损失
        self.stride = strides[layer_no]  # 步长
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]  # 所有的锚点网格
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]  # 掩码后的锚点
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))  # 参考锚点
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)  # 设置参考锚点的宽和高
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)  # 将参考锚点转换为浮点张量
        # 添加卷积层，其中输入通道数为in_ch，输出通道数为锚点数量乘以（类别数量+5），卷积核大小为1，步长为1，不添加偏置项
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)
def forward(self, xin, labels=None):
    """
    前向传播函数。
    Args:
        xin (torch.Tensor): 输入特征图，大小为 :math:`(N, C, H, W)`，其中 N, C, H, W 分别表示批次大小，通道宽度，高度，宽度。
        labels (torch.Tensor): 标签数据，大小为 :math:`(N, K, 5)`。N 和 K 分别表示批次大小和标签数量。
            每个标签由 [class, xc, yc, w, h] 组成：
                class (float): 类别索引。
                xc, yc (float) : 边界框的中心，值范围从0到1。
                w, h (float) : 边界框的大小，值范围从0到1。
    Returns:
        loss (torch.Tensor): 总损失 - 反向传播的目标。
        loss_xy (torch.Tensor): x, y 损失 - 通过具有框大小依赖权重的二元交叉熵 (BCE) 计算。
        loss_wh (torch.Tensor): w, h 损失 - 通过 l2 计算，没有尺寸平均，具有框大小依赖权重。
        loss_obj (torch.Tensor): 对象性损失 - 通过 BCE 计算。
        loss_cls (torch.Tensor): 分类损失 - 对于每个类别，通过 BCE 计算。
        loss_l2 (torch.Tensor): 总 l2 损失 - 仅用于记录。
    """
    output = self.conv(xin)  # 通过卷积层传递输入

    batchsize = output.shape[0]  # 获取批次大小
    fsize = output.shape[2]  # 获取特征图的大小
    n_ch = 5 + self.n_classes  # 计算通道数
    dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor  # 根据输入是否在CUDA上确定数据类型

    output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)  # 改变输出的形状
    output = output.permute(0, 1, 3, 4, 2)  # 改变维度的顺序

    # 对 xy, obj, cls 使用 logistic 激活函数
    output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
        output[..., np.r_[:2, 4:n_ch]])

    # 计算预测 - xywh obj cls

    x_shift = dtype(np.broadcast_to(
        np.arange(fsize, dtype=np.float32), output.shape[:4]))  # 创建一个与输出形状相同的x偏移量
    y_shift = dtype(np.broadcast_to(
        np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))  # 创建一个与输出形状相同的y偏移量

    masked_anchors = np.array(self.masked_anchors)

    w_anchors = dtype(np.broadcast_to(np.reshape(
        masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))  # 创建一个与输出形状相同的宽度锚点
    h_anchors = dtype(np.broadcast_to(np.reshape(
        masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))  # 创建一个与输出形状相同的高度锚点

    pred = output.clone()  # 克隆输出
    pred[..., 0] += x_shift  # 更新预测的x坐标
    pred[..., 1] += y_shift  # 更新预测的y坐标
    pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors  # 更新预测的宽度
    pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors  # 更新预测的高度

    if labels is None:  # 不在训练中
        pred[..., :4] *= self.stride
        return pred.view(batchsize, -1, n_ch).data

    pred = pred[..., :4].data

    # 目标分配

    tgt_mask = torch.zeros(batchsize, self.n_anchors,
                           fsize, fsize, 4 + self.n_classes).type(dtype)  # 创建目标掩码
    obj_mask = torch.ones(batchsize, self.n_anchors,
                          fsize, fsize).type(dtype)  # 创建对象掩码
    tgt_scale = torch.zeros(batchsize, self.n_anchors,
                            fsize, fsize, 2).type(dtype)  # 创建目标尺度

    target = torch.zeros(batchsize, self.n_anchors,
                         fsize, fsize, n_ch).type(dtype)  # 创建目标

    labels = labels.cpu().data  # 将标签转移到CPU上
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # 计算对象的数量

    truth_x_all = labels[:, :, 1] * fsize  # 计算所有真实的x坐标
    truth_y_all = labels[:, :, 2] * fsize  # 计算所有真实的y坐标
    truth_w_all = labels[:, :, 3] * fsize  # 计算所有真实的宽度
    truth_h_all = labels[:, :, 4] * fsize  # 计算所有真实的高度
    truth_i_all = truth_x_all.to(torch.int16).numpy()  # 将所有真实的x坐标转换为整数
    truth_j_all = truth_y_all.to(torch.int16).numpy()  # 将所有真实的y坐标转换为整数
    for b in range(batchsize):  # 对每个批次进行循环
        n = int(nlabel[b])  # 获取标签的数量
        if n == 0:  # 如果没有标签，跳过这个批次
            continue
        truth_box = dtype(np.zeros((n, 4)))  # 创建一个空的真实框数组
        truth_box[:n, 2] = truth_w_all[b, :n]  # 设置真实框的宽度
        truth_box[:n, 3] = truth_h_all[b, :n]  # 设置真实框的高度
        truth_i = truth_i_all[b, :n]  # 获取真实框的i坐标
        truth_j = truth_j_all[b, :n]  # 获取真实框的j坐标
    
        # 计算真实框和参考锚点之间的iou
        anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
        best_n_all = np.argmax(anchor_ious_all, axis=1)  # 找到iou最大的锚点
        best_n = best_n_all % 3  # 获取最佳锚点的索引
        best_n_mask = ((best_n_all == self.anch_mask[0]) | (
            best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))  # 创建一个最佳锚点的掩码
    
        truth_box[:n, 0] = truth_x_all[b, :n]  # 设置真实框的x坐标
        truth_box[:n, 1] = truth_y_all[b, :n]  # 设置真实框的y坐标
    
        # 计算预测框和真实框之间的iou
        pred_ious = bboxes_iou(
            pred[b].view(-1, 4), truth_box, xyxy=False)
        pred_best_iou, _ = pred_ious.max(dim=1)  # 找到iou最大的预测框
        pred_best_iou = (pred_best_iou > self.ignore_thre)  # 创建一个掩码，忽略iou小于阈值的预测框
        pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
        # 如果预测框匹配真实框，将掩码设置为零（忽略）
        obj_mask[b] = 1 - pred_best_iou
    
        if sum(best_n_mask) == 0:  # 如果没有最佳锚点，跳过这个批次
            continue
        
        for ti in range(best_n.shape[0]):  # 对每个最佳锚点进行循环
            if best_n_mask[ti] == 1:  # 如果这个锚点是最佳锚点
                i, j = truth_i[ti], truth_j[ti]  # 获取真实框的坐标
                a = best_n[ti]  # 获取最佳锚点的索引
                obj_mask[b, a, j, i] = 1  # 将掩码设置为1（不忽略）
                tgt_mask[b, a, j, i, :] = 1  # 将目标掩码设置为1
                target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                    truth_x_all[b, ti].to(torch.int16).to(torch.float)  # 计算目标x坐标的偏移量
                target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                    truth_y_all[b, ti].to(torch.int16).to(torch.float)  # 计算目标y坐标的偏移量
                target[b, a, j, i, 2] = torch.log(
                    truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)  # 计算目标宽度的对数变换
                target[b, a, j, i, 3] = torch.log(
                    truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)  # 计算目标高度的对数变换
                target[b, a, j, i, 4] = 1  # 将目标存在的概率设置为1
                target[b, a, j, i, 5 + labels[b, ti,
                                              0].to(torch.int16).numpy()] = 1  # 将目标类别设置为1
                tgt_scale[b, a, j, i, :] = torch.sqrt(
                    2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)  # 计算目标的缩放因子
    
    # 计算损失
    
    output[..., 4] *= obj_mask  # 应用掩码到预测的存在概率
    output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask  # 应用目标掩码到预测的坐标和类别
    output[..., 2:4] *= tgt_scale  # 应用缩放因子到预测的宽度和高度
    
    target[..., 4] *= obj_mask  # 应用掩码到目标的存在概率
    target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask  # 应用目标掩码到目标的坐标和类别
    target[..., 2:4] *= tgt_scale  # 应用缩放因子到目标的宽度和高度
    
    bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale,
                         size_average=False)  # 计算加权的二元交叉熵损失
    loss_xy = bceloss(output[..., :2], target[..., :2])  # 计算坐标的损失
    loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2  # 计算宽度和高度的损失
    loss_obj = self.bce_loss(output[..., 4], target[..., 4])  # 计算存在概率的损失
    loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])  # 计算类别的损失
    loss_l2 = self.l2_loss(output, target)  # 计算L2损失
    
    loss = loss_xy + loss_wh + loss_obj + loss_cls  # 计算总损失
    
    return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2  # 返回总损失和各部分损失
        