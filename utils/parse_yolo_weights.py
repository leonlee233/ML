from __future__ import division
import torch
import numpy as np

def parse_conv_block(m, weights, offset, initflag):
    """
    初始化带有批量归一化的卷积层
    Args:
        m (Sequential): 层序列
        weights (numpy.ndarray): 预训练的权重数据
        offset (int): 当前在权重文件中的位置
        initflag (bool): 如果为True，则该层不在权重文件中覆盖。\
            它们使用darknet风格的初始化进行初始化。
    Returns:
        offset (int): 当前在权重文件中的位置
        weights (numpy.ndarray): 预训练的权重数据
    """
    conv_model = m[0]  # 获取卷积模型
    bn_model = m[1]  # 获取批量归一化模型
    param_length = m[1].bias.numel()  # 获取偏置项的数量

    # 批量归一化
    for pname in ['bias', 'weight', 'running_mean', 'running_var']:
        layerparam = getattr(bn_model, pname)  # 获取批量归一化模型的参数

        if initflag:  # yolo初始化 - 缩放到一，偏置为零
            if pname == 'weight':
                weights = np.append(weights, np.ones(param_length))  # 权重初始化为1
            else:
                weights = np.append(weights, np.zeros(param_length))  # 偏置、均值和方差初始化为0

        param = torch.from_numpy(weights[offset:offset + param_length]).view_as(layerparam)  # 将权重转换为张量
        layerparam.data.copy_(param)  # 复制参数到层参数
        offset += param_length  # 更新偏移量

    param_length = conv_model.weight.numel()  # 获取卷积权重的数量

    # 卷积
    if initflag:  # yolo初始化
        n, c, k, _ = conv_model.weight.shape  # 获取卷积权重的形状
        scale = np.sqrt(2 / (k * k * c))  # 计算缩放因子
        weights = np.append(weights, scale * np.random.normal(size=param_length))  # 初始化卷积权重

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)  # 将权重转换为张量
    conv_model.weight.data.copy_(param)  # 复制参数到卷积权重
    offset += param_length  # 更新偏移量

    return offset, weights  # 返回当前偏移量和权重

def parse_yolo_block(m, weights, offset, initflag):
    """
    YOLO Layer (one conv with bias) Initialization
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    # 获取卷积模型
    conv_model = m._modules['conv']
    # 获取偏置参数的数量
    param_length = conv_model.bias.numel()

    # 如果是初始化阶段，将偏置参数全部设为0
    if initflag: 
        weights = np.append(weights, np.zeros(param_length))

    # 从权重数据中获取偏置参数，并将其形状调整为与模型中的偏置参数相同
    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.bias)
    # 将获取的偏置参数复制到模型中
    conv_model.bias.data.copy_(param)
    # 更新当前在权重文件中的位置
    offset += param_length

    # 获取权重参数的数量
    param_length = conv_model.weight.numel()

    # 如果是初始化阶段，使用darknet风格的初始化方法初始化权重参数
    if initflag: 
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))
 
    # 从权重数据中获取权重参数，并将其形状调整为与模型中的权重参数相同
    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    # 将获取的权重参数复制到模型中
    conv_model.weight.data.copy_(param)
    # 更新当前在权重文件中的位置
    offset += param_length

    # 返回当前在权重文件中的位置和权重数据
    return offset, weights


def parse_yolo_weights(model, weights_path):
    """
    解析YOLO（darknet）预训练的权重数据到pytorch模型
    Args:
        model : pytorch模型对象
        weights_path (str): YOLO（darknet）预训练权重文件的路径
    """
    fp = open(weights_path, "rb")  # 打开权重文件

    # 跳过头部
    header = np.fromfile(fp, dtype=np.int32, count=5)  # 不使用
    # 读取权重
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()  # 关闭文件

    offset = 0 
    initflag = False  # 整个yolo权重：False，darknet权重：True

    for m in model.module_list:  # 遍历模型的每一层

        if m._get_name() == 'Sequential':
            # 正常的卷积块
            offset, weights = parse_conv_block(m, weights, offset, initflag)

        elif m._get_name() == 'resblock':
            # 残差块
            for modu in m._modules['module_list']:
                for blk in modu:
                    offset, weights = parse_conv_block(blk, weights, offset, initflag)

        elif m._get_name() == 'YOLOLayer':
            # YOLO层（一个带偏置的卷积）初始化
            offset, weights = parse_yolo_block(m, weights, offset, initflag)

        initflag = (offset >= len(weights))  # 权重文件的末尾。打开标志
