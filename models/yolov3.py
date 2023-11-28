import torch
import torch.nn as nn

from collections import defaultdict
from models.yolo_layer import YOLOLayer

def add_conv(in_ch, out_ch, ksize, stride):
    """
    添加一个conv2d / batchnorm / leaky ReLU块。
    Args:
        in_ch (int): 卷积层的输入通道数。
        out_ch (int): 卷积层的输出通道数。
        ksize (int): 卷积层的核大小。
        stride (int): 卷积层的步长。
    Returns:
        stage (Sequential) : 组成卷积块的顺序层。
    """
    stage = nn.Sequential()  # 创建一个Sequential对象，它是一个包含多个网络层的容器
    pad = (ksize - 1) // 2  # 计算填充大小，这是为了保证卷积后的特征图大小不变
    # 添加卷积层，其中in_channels是输入通道数，out_channels是输出通道数，
    # kernel_size是卷积核大小，stride是步长，padding是填充大小，bias=False表示不添加偏置项
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    # 添加批量归一化层，其中out_ch是特征图的通道数
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    # 添加LeakyReLU激活函数，其中0.1是负轴区域的斜率
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage  # 返回包含卷积、批量归一化和LeakyReLU的网络块

class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):
        """
        初始化方法
        Args:
            ch (int): 输入和输出通道的数量。
            nblocks (int): 残差块的数量。
            shortcut (bool): 如果为True，则启用残差张量加法。
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置是否启用残差张量加法
        self.shortcut = shortcut
        # 创建一个模块列表来存储残差块
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            # 对于每一个残差块，我们创建一个包含两个卷积层的模块列表
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch//2, 1, 1))
            resblock_one.append(add_conv(ch//2, ch, 3, 1))
            # 将这个残差块添加到模块列表中
            self.module_list.append(resblock_one)

    def forward(self, x):
        """
        前向传播方法
        Args:
            x (Tensor): 输入张量
        Returns:
            x (Tensor): 输出张量
        """
        # 对于模块列表中的每一个残差块
        for module in self.module_list:
            # 我们首先复制输入张量
            h = x
            # 然后对复制的张量进行卷积操作
            for res in module:
                h = res(h)
            # 如果启用了残差张量加法，我们将卷积后的张量与输入张量相加
            # 否则，我们直接使用卷积后的张量
            x = x + h if self.shortcut else h
        # 返回输出张量
        return x
def create_yolov3_modules(config_model, ignore_thre):
    """
    创建YOLOv3层模块。
    Args:
        config_model (dict): 模型配置。详见YOLOLayer类。
        ignore_thre (float): 在YOLOLayer中使用。
    Returns:
        mlist (ModuleList): YOLOv3模块列表。
    """

    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))  # 添加卷积层，输入通道数为3，输出通道数为32，卷积核大小为3，步长为1
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))  # 添加卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为2
    mlist.append(resblock(ch=64))  # 添加残差块，通道数为64
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))  # 添加卷积层，输入通道数为64，输出通道数为128，卷积核大小为3，步长为2
    mlist.append(resblock(ch=128, nblocks=2))  # 添加残差块，通道数为128，块数为2
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))  # 添加卷积层，输入通道数为128，输出通道数为256，卷积核大小为3，步长为2
    mlist.append(resblock(ch=256, nblocks=8))    # 从这里开始shortcut 1，添加残差块，通道数为256，块数为8
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))  # 添加卷积层，输入通道数为256，输出通道数为512，卷积核大小为3，步长为2
    mlist.append(resblock(ch=512, nblocks=8))    # 从这里开始shortcut 2，添加残差块，通道数为512，块数为8
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))  # 添加卷积层，输入通道数为512，输出通道数为1024，卷积核大小为3，步长为2
    mlist.append(resblock(ch=1024, nblocks=4))  # 添加残差块，通道数为1024，块数为4

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))  # 添加残差块，通道数为1024，块数为2，不使用shortcut
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 添加卷积层，输入通道数为1024，输出通道数为512，卷积核大小为1，步长为1
    # 第一个yolo分支
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 添加卷积层，输入通道数为512，输出通道数为1024，卷积核大小为3，步长为1
    mlist.append(
         YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thre))  # 添加YOLO层，模型配置为config_model，层号为0，输入通道数为1024，忽略阈值为ignore_thre

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 添加卷积层，输入通道数为512，输出通道数为256，卷积核大小为1，步长为1
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 添加上采样层，缩放因子为2，模式为'nearest'
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))  # 添加卷积层，输入通道数为768，输出通道数为256，卷积核大小为1，步长为1
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 添加卷积层，输入通道数为256，输出通道数为512，卷积核大小为3，步长为1
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))  # 添加残差块，通道数为512，块数为1，不使用shortcut
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 添加卷积层，输入通道数为512，输出通道数为256，卷积核大小为1，步长为1
    # 第二个yolo分支
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 添加卷积层，输入通道数为256，输出通道数为512，卷积核大小为3，步长为1
    mlist.append(
        YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thre))  # 添加YOLO层，模型配置为config_model，层号为1，输入通道数为512，忽略阈值为ignore_thre

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 添加卷积层，输入通道数为256，输出通道数为128，卷积核大小为1，步长为1
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 添加上采样层，缩放因子为2，模式为'nearest'
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))  # 添加卷积层，输入通道数为384，输出通道数为128，卷积核大小为1，步长为1
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 添加卷积层，输入通道数为128，输出通道数为256，卷积核大小为3，步长为1
    mlist.append(resblock(ch=256, nblocks=2, shortcut=False))  # 添加残差块，通道数为256，块数为2，不使用shortcut
    mlist.append(
         YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre))  # 添加YOLO层，模型配置为config_model，层号为2，输入通道数为256，忽略阈值为ignore_thre

    return mlist  # 返回模块列表


class YOLOv3(nn.Module):
    """
    YOLOv3模型模块。模块列表由create_yolov3_modules函数定义。
    在训练期间，网络从三个YOLO层返回损失值，在测试期间返回检测结果。
    """
    def __init__(self, config_model, ignore_thre=0.7):
        """
        YOLOv3类的初始化。
        Args:
            config_model (dict): 在YOLOLayer中使用。
            ignore_thre (float): 在YOLOLayer中使用。
        """
        super(YOLOv3, self).__init__()  # 调用父类nn.Module的初始化方法

        if config_model['TYPE'] == 'YOLOv3':
            self.module_list = create_yolov3_modules(config_model, ignore_thre)  # 创建YOLOv3模块
        else:
            raise Exception('Model name {} is not available'.format(config_model['TYPE']))  # 如果模型类型不是YOLOv3，抛出异常

    def forward(self, x, targets=None):
        """
        YOLOv3的前向传播路径。
        Args:
            x (torch.Tensor) : 输入数据，其形状为(N, C, H, W)，其中N, C分别为批大小和通道数。
            targets (torch.Tensor) : 标签数组，其形状为(N, 50, 5)

        Returns:
            训练时：
                output (torch.Tensor): 用于反向传播的损失张量。
            测试时：
                output (torch.Tensor): 连接的检测结果。
        """
        train = targets is not None  # 判断是否为训练模式
        output = []  # 存储输出的列表
        self.loss_dict = defaultdict(float)  # 存储损失的字典
        route_layers = []  # 存储路由层的列表
        for i, module in enumerate(self.module_list):  # 遍历模块列表
            # yolo层
            if i in [14, 22, 28]:
                if train:  # 如果是训练模式
                    x, *loss_dict = module(x, targets)  # 获取损失
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):  # 遍历损失字典
                        self.loss_dict[name] += loss  # 累加损失
                else:  # 如果是测试模式
                    x = module(x)  # 获取输出
                output.append(x)  # 将输出添加到列表中
            else:  # 如果不是yolo层
                x = module(x)  # 获取输出

            # 路由层
            if i in [6, 8, 12, 20]:
                route_layers.append(x)  # 将输出添加到路由层列表中
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo第二层
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)  # 沿着通道维度连接输出和路由层
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)  # 沿着通道维度连接输出和路由层
        if train:  # 如果是训练模式
            return sum(output)  # 返回损失的和
        else:  # 如果是测试模式
            return torch.cat(output, 1)  # 返回连接的输出
