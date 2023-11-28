import json
import tempfile

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from dataset.cocodataset import *
from utils.utils import *

class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed 
    and evaluated by COCO API.
    """
    def __init__(self, model_type, data_dir, img_size, confthre, nmsthre):
        """
        初始化方法
        Args:
            model_type (str): 在配置文件中指定的模型名称
            data_dir (str): 数据集的根目录
            img_size (int): 预处理后的图像大小。图像被调整为形状为(img_size, img_size)的正方形。
            confthre (float): 范围从0到1的置信度阈值，该阈值在配置文件中定义。
            nmsthre (float): 非最大抑制的IoU阈值，范围从0到1。
        """

        # 数据增强设置
        augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                        'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}

        # 创建COCODataset对象
        self.dataset = COCODataset(model_type=model_type,
                                   data_dir=data_dir,
                                   img_size=img_size,
                                   augmentation=augmentation,
                                   json_file='instances_val2017.json',
                                   name='val2017')
        # 创建数据加载器
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0)
        # 设置图像大小
        self.img_size = img_size
        # 设置置信度阈值（来自darknet）
        self.confthre = 0.005 
        # 设置非最大抑制的IoU阈值（来自darknet）
        self.nmsthre = nmsthre 

    def evaluate(self, model):
        """
        COCO平均精度（AP）评估。在测试数据集上进行迭代推理，
        并通过COCO API进行结果评估。
        Args:
            model : 模型对象
        Returns:
            ap50_95 (float) : 计算的COCO AP，IoU=50:95
            ap50 (float) : 计算的COCO AP，IoU=50
        """
        model.eval()  # 将模型设置为评估模式
        cuda = torch.cuda.is_available()  # 检查CUDA是否可用
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # 根据CUDA是否可用选择适当的张量类型
        ids = []  # 用于存储图像ID的列表
        data_dict = []  # 用于存储数据字典的列表
        dataiterator = iter(self.dataloader)  # 创建数据加载器的迭代器
        while True:  # 遍历val2017中的所有数据
            try:
                img, _, info_img, id_ = next(dataiterator)  # 加载一个批次
            except StopIteration:
                break  # 如果没有更多的数据，就跳出循环
            info_img = [float(info) for info in info_img]  # 将信息图像转换为浮点数列表
            id_ = int(id_)  # 将ID转换为整数
            ids.append(id_)  # 将ID添加到列表中
            with torch.no_grad():  # 不计算梯度
                img = Variable(img.type(Tensor))  # 将图像转换为变量
                outputs = model(img)  # 通过模型获取输出
                outputs = postprocess(
                    outputs, 80, self.confthre, self.nmsthre)  # 对输出进行后处理
                if outputs[0] is None:
                    continue  # 如果输出为空，就跳过这个批次
                outputs = outputs[0].cpu().data  # 将输出转换为CPU数据

            for output in outputs:  # 遍历输出
                x1 = float(output[0])  # 获取x1坐标
                y1 = float(output[1])  # 获取y1坐标
                x2 = float(output[2])  # 获取x2坐标
                y2 = float(output[3])  # 获取y2坐标
                label = self.dataset.class_ids[int(output[6])]  # 获取标签
                box = yolobox2label((y1, x1, y2, x2), info_img)  # 将YOLO框转换为标签
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]  # 计算边界框
                score = float(output[4].data.item() * output[5].data.item())  # 计算得分（对象得分 * 类别得分）
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score, "segmentation": []}  # 创建COCO json格式的数据
                data_dict.append(A)  # 将数据添加到列表中

        annType = ['segm', 'bbox', 'keypoints']  # 注释类型

        # 评估Dt（检测）json与ground truth的比较
        if len(data_dict) > 0:
            # 获取数据集的COCO对象
            cocoGt = self.dataset.coco
            # 临时解决方案：因为pycocotools在py36中无法处理字典，所以暂时将数据写入json文件。
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            # 加载检测结果
            cocoDt = cocoGt.loadRes(tmp)
            # 创建COCOeval对象，用于评估检测结果
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            # 设置需要评估的图像ID
            cocoEval.params.imgIds = ids
            # 进行评估
            cocoEval.evaluate()
            # 累积评估结果
            cocoEval.accumulate()
            # 总结评估结果
            cocoEval.summarize()
            # 返回评估统计数据的前两项
            return cocoEval.stats[0], cocoEval.stats[1]
        else:
            # 如果data_dict为空，则返回0, 0
            return 0, 0
