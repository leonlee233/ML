import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from utils.utils import *

class COCODataset(Dataset):  # 定义一个名为COCODataset的类，它继承了Dataset类
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir='COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=416,
                 augmentation=None, min_size=1, debug=False):  # 类的初始化函数
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir  # 数据集的根目录
        self.json_file = json_file  # COCO json文件名
        self.model_type = model_type  # 模型类型
        self.coco = COCO(self.data_dir+'annotations/'+self.json_file)  # 使用COCO API读取注释数据
        self.ids = self.coco.getImgIds()  # 获取图像的id
        if debug:  # 如果处于调试模式
            self.ids = self.ids[1:2]  # 只选择一个数据id
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())  # 获取并排序类别id
        self.name = name  # 数据集的名称
        self.max_labels = 50  # 最大标签数
        self.img_size = img_size  # 图像的目标大小
        self.min_size = min_size  # 忽略小于此大小的边界框
        self.lrflip = augmentation['LRFLIP']  # 是否进行左右翻转
        self.jitter = augmentation['JITTER']  # 是否进行抖动
        self.random_placing = augmentation['RANDOM_PLACING']  # 是否进行随机放置
        self.hue = augmentation['HUE']  # 色调的变化范围
        self.saturation = augmentation['SATURATION']  # 饱和度的变化范围
        self.exposure = augmentation['EXPOSURE']  # 曝光度的变化范围
        self.random_distort = augmentation['RANDOM_DISTORT']  # 是否进行随机扭曲

    def __len__(self):  # 定义一个名为__len__的函数，它返回数据集的长度
        return len(self.ids)  # 返回图像id的数量
    def __getitem__(self, index):
        """
        为给定的索引选取一个图像/标签对并进行预处理。
        Args:
            index (int): 数据索引
        Returns:
            img (numpy.ndarray): 预处理后的图像
            padded_labels (torch.Tensor): 预处理后的标签数据。形状为 :math:`[self.max_labels, 5]`。每个标签由 [class, xc, yc, w, h] 组成：
                class (float): 类别索引。
                xc, yc (float) : 边界框的中心，值范围从0到1。
                w, h (float) : 边界框的大小，值范围从0到1。
            info_img : 元组，包含 h, w, nh, nw, dx, dy。
                h, w (int): 图像的原始形状
                nh, nw (int): 无填充的调整后图像的形状
                dx, dy (int): 填充大小
            id_ (int): 与输入索引相同。用于评估。
        """
        id_ = self.ids[index]  # 获取索引对应的id
    
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)  # 获取注释id
        annotations = self.coco.loadAnns(anno_ids)  # 加载注释
    
        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:  # 随机决定是否进行左右翻转
            lrflip = True
    
        # 加载图像并进行预处理
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')  # 构造图像文件路径
        img = cv2.imread(img_file)  # 读取图像文件
    
        if self.json_file == 'instances_val5k.json' and img is None:  # 如果图像不存在，则尝试在另一个文件夹中查找
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)
        assert img is not None  # 确保图像存在
    
        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)  # 对图像进行预处理
    
        if self.random_distort:  # 如果启用了随机扭曲，则对图像进行随机扭曲
            img = random_distort(img, self.hue, self.saturation, self.exposure)
    
        img = np.transpose(img / 255., (2, 0, 1))  # 对图像进行归一化和转置
    
        if lrflip:  # 如果进行了左右翻转，则对图像进行翻转
            img = np.flip(img, axis=2).copy()
    
        # 加载标签
        labels = []
        for anno in annotations:  # 遍历所有注释
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:  # 如果边界框的大小大于最小尺寸，则添加到标签中
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))  # 添加类别索引
                labels[-1].extend(anno['bbox'])  # 添加边界框
    
        padded_labels = np.zeros((self.max_labels, 5))  # 创建填充标签
        if len(labels) > 0:  # 如果有标签
            labels = np.stack(labels)  # 将标签堆叠成数组
            if 'YOLO' in self.model_type:  # 如果模型类型为YOLO
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)  # 将标签转换为YOLO框
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]  # 将标签添加到填充标签中
        padded_labels = torch.from_numpy(padded_labels)  # 将填充标签转换为张量
    
        return img, padded_labels, info_img, id_  # 返回图像，填充标签，图像信息和id
