from __future__ import division
import torch
import numpy as np
import cv2

def nms(bbox, thresh, score=None, limit=None):
    # 如果没有边界框，则返回一个空数组
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    # 按分数对边界框进行排序（如果提供了分数）
    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]

    # 计算每个边界框的面积
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    # 初始化所选边界框的列表
    selec = np.zeros(bbox.shape[0], dtype=bool)

    # 遍历每个边界框
    for i, b in enumerate(bbox):
        # 计算当前边界框与所选边界框之间的重叠
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        # 计算当前边界框与所选边界框之间的IoU
        iou = area / (bbox_area[i] + bbox_area[selec] - area)

        # 如果IoU大于阈值，则跳过此边界框
        if (iou >= thresh).any():
            continue

        # 否则，选择此边界框
        selec[i] = True

        # 如果我们已经达到了限制，则停止迭代
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    # 获取所选边界框的索引
    selec = np.where(selec)[0]

    # 按分数对索引进行排序（如果提供了分数）
    if score is not None:
        selec = order[selec]

    # 返回所选索引
    return selec.astype(np.int32)

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    YOLO模型输出的后处理
    执行边界框变换，为每个检测指定类别，并执行类别级的非极大值抑制。
    参数:
        prediction (torch tensor): 形状为(N, B, 4)。
            N是预测的数量，B是边界框的数量。最后一个轴由xc, yc, w, h组成，
            其中xc和yc表示边界框的中心。
        num_classes (int): 数据集类别的数量。
        conf_thre (float): 范围从0到1的置信度阈值，定义在配置文件中。
        nms_thre (float): 非极大值抑制的IoU阈值，范围从0到1。

    返回:
        output (torch tensor列表):
    """
    # 创建一个新的tensor，形状与prediction相同
    box_corner = prediction.new(prediction.shape)
    # 计算边界框的四个角的坐标
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # 更新prediction的前四个通道为边界框的坐标
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 初始化输出列表
    output = [None for _ in range(len(prediction))]
    # 遍历每个预测
    for i, image_pred in enumerate(prediction):
        # 过滤出低于阈值的置信度分数
        class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        class_pred = class_pred[0]
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # 如果没有剩余 => 处理下一张图像
        if not image_pred.size(0):
            continue
        # 获取置信度分数高于阈值的检测
        ind = (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre).nonzero()
        # 按照(x1, y1, x2, y2, obj_conf, class_conf, class_pred)的顺序排列检测
        detections = torch.cat((
                image_pred[ind[:, 0], :5],
                image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
                ind[:, 1].float().unsqueeze(1)
                ), 1)
        # 遍历所有预测的类别
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # 获取特定类别的检测
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4]*nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """
    计算边界框的交并比（IoU）。
    IoU是交集区域和并集区域的面积之比。

    参数:
        bboxes_a (array): 形状为(N, 4)的数组。
            N是边界框的数量。数据类型应为numpy.float32。
        bboxes_b (array): 形状为(K, 4)的数组，与bboxes_a类似。
            数据类型应为numpy.float32。
        xyxy (bool): 如果为True，则边界框的格式为(x1, y1, x2, y2)，
            否则为(xc, yc, w, h)，其中xc和yc为中心坐标，w和h为宽和高。

    返回:
        array: 形状为(N, K)的数组。
            索引(n, k)处的元素包含了bboxes_a中第n个边界框和bboxes_b中第k个边界框的IoU。

    来源: https://github.com/chainer/chainercv
    """
    # 检查输入边界框的形状是否正确
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # 计算交集的左上角坐标
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])  # 左上角坐标
        # 计算交集的右下角坐标
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])  # 右下角坐标
        # 计算两组边界框的面积
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # 计算交集的右下角坐标
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # 计算两组边界框的面积
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    # 计算交集的面积
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en

    # 计算并返回IoU
    return area_i / (area_a[:, None] + area_b - area_i)


def label2yolobox(labels, info_img, maxsize, lrflip):
    """
    将COCO标签转换为YOLO边界框标签
    参数:
        labels (numpy.ndarray): 形状为(N, 5)的标签数据。
            每个标签由[class, x, y, w, h]组成，其中
                class (float): 类别索引。
                x, y, w, h (float): 左上角点的坐标，边界框的宽度和高度。
                    值的范围从0到图像的宽度或高度。
        info_img: 包含h, w, nh, nw, dx, dy的元组。
            h, w (int): 图像的原始形状
            nh, nw (int): 无填充的调整后图像的形状
            dx, dy (int): 填充大小
        maxsize (int): 预处理后的目标图像大小
        lrflip (bool): 水平翻转标志

    返回:
        labels: 形状为(N, 5)的标签数据。
            每个标签由[class, xc, yc, w, h]组成，其中
                class (float): 类别索引。
                xc, yc (float): 边界框中心的坐标，值的范围从0到1。
                w, h (float): 边界框的大小，值的范围从0到1。
    """
    # 解包图像信息
    h, w, nh, nw, dx, dy = info_img
    # 计算边界框的左上角和右下角坐标
    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h
    # 更新标签的坐标和大小
    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize
    # 如果进行了水平翻转，更新x坐标
    if lrflip:
        labels[:, 1] = 1 - labels[:, 1]
    return labels


def yolobox2label(box, info_img):
    """
    将YOLO边界框标签转换为yxyx边界框标签。
    参数:
        box (list): 格式为[yc, xc, w, h]的边界框数据，
            坐标系为预处理后的坐标系。
        info_img: 包含h, w, nh, nw, dx, dy的元组。
            h, w (int): 图像的原始形状
            nh, nw (int): 无填充的调整后图像的形状
            dx, dy (int): 填充大小
    返回:
        label (list): 格式为[y1, x1, y2, x2]的边界框数据，
            坐标系为输入图像的坐标系。
    """
    # 解包图像信息
    h, w, nh, nw, dx, dy = info_img
    y1, x1, y2, x2 = box
    # 计算边界框的高度和宽度
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    # 计算边界框的左上角坐标
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    # 创建并返回标签
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label


def preprocess(img, imgsize, jitter, random_placing=False):
    """
    对YOLO输入进行图像预处理
    对图像的较短边进行填充，并调整大小为(imgsize, imgsize)
    参数:
        img (numpy.ndarray): 形状为(H, W, C)的输入图像。
            值的范围从0到255。
        imgsize (int): 预处理后的目标图像大小
        jitter (float): 调整大小的抖动幅度
        random_placing (bool): 如果为True，将图像放在随机位置

    返回:
        img (numpy.ndarray): 形状为(C, imgsize, imgsize)的输入图像。
            值的范围从0到1。
        info_img: 包含h, w, nh, nw, dx, dy的元组。
            h, w (int): 图像的原始形状
            nh, nw (int): 无填充的调整后图像的形状
            dx, dy (int): 填充大小
    """
    # 获取图像的形状
    h, w, _ = img.shape
    # 将图像的颜色空间从RGB转换为BGR
    img = img[:, :, ::-1]
    assert img is not None

    # 如果设置了抖动，计算新的宽高比
    if jitter > 0:
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw))\
                 / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h

    # 根据新的宽高比计算调整后的宽度和高度
    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    # 如果设置了随机放置，计算填充的大小
    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    # 调整图像的大小
    img = cv2.resize(img, (nw, nh))
    # 创建一个新的图像，并将调整大小后的图像放在其中
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy+nh, dx:dx+nw, :] = img

    # 创建并返回图像信息
    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img


def rand_scale(s):
    """
    计算随机缩放因子
    参数:
        s (float): 随机缩放的范围。
    返回:
        随机缩放因子 (float)，其范围从1 / s到s。
    """
    # 在1到s之间均匀地生成一个随机数
    scale = np.random.uniform(low=1, high=s)
    # 如果生成的随机数大于0.5，返回scale，否则返回1 / scale
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def random_distort(img, hue, saturation, exposure):
    """
    在HSV颜色空间中进行随机扭曲。
    参数:
        img (numpy.ndarray): 形状为(H, W, C)的输入图像。
            值的范围从0到255。
        hue (float): 随机扭曲参数。
        saturation (float): 随机扭曲参数。
        exposure (float): 随机扭曲参数。
    返回:
        img (numpy.ndarray)
    """
    # 在-hue到hue之间均匀地生成一个随机数
    dhue = np.random.uniform(low=-hue, high=hue)
    # 计算随机缩放因子
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    # 将图像的颜色空间从RGB转换为HSV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 将图像的数据类型转换为float32，并将值的范围从0-255转换为0-1
    img = np.asarray(img, dtype=np.float32) / 255.
    # 调整图像的饱和度和亮度
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    # 调整图像的色调
    H = img[:, :, 0] + dhue

    # 如果dhue大于0，将H中大于1.0的值减去1.0
    if dhue > 0:
        H[H > 1.0] -= 1.0
    # 如果dhue小于0，将H中小于0.0的值加上1.0
    else:
        H[H < 0.0] += 1.0

    # 更新图像的色调
    img[:, :, 0] = H
    # 将图像的值的范围从0-1转换为0-255，并将数据类型转换为uint8
    img = (img * 255).clip(0, 255).astype(np.uint8)
    # 将图像的颜色空间从HSV转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # 将图像的数据类型转换为float32
    img = np.asarray(img, dtype=np.float32)

    return img

def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors

