import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights

def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    parser = argparse.ArgumentParser()  # 创建一个命令行参数解析器
    parser.add_argument('--gpu', type=int, default=0)  # 添加一个命令行参数，指定使用哪个GPU
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')  # 添加一个命令行参数，指定配置文件的路径
    parser.add_argument('--ckpt', type=str,
                        help='path to the checkpoint file')  # 添加一个命令行参数，指定检查点文件的路径
    parser.add_argument('--weights_path', type=str,
                        default=None, help='path to weights file')  # 添加一个命令行参数，指定权重文件的路径
    parser.add_argument('--image', type=str)  # 添加一个命令行参数，指定要进行检测的图像的路径
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./output.png")')  # 添加一个命令行参数，指定是否在后台模式下运行
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')  # 添加一个命令行参数，指定置信度阈值
    args = parser.parse_args()  # 解析命令行参数

    with open(args.cfg, 'r') as f:  # 打开配置文件
        cfg = yaml.load(f)  # 加载配置文件

    imgsize = cfg['TEST']['IMGSIZE']  # 获取图像大小
    model = YOLOv3(cfg['MODEL'])  # 创建YOLOv3模型

    confthre = cfg['TEST']['CONFTHRE']  # 获取置信度阈值
    nmsthre = cfg['TEST']['NMSTHRE']  # 获取非极大值抑制阈值

    if args.detect_thresh:  # 如果指定了置信度阈值
        confthre = args.detect_thresh  # 使用指定的置信度阈值

    img = cv2.imread(args.image)  # 读取图像
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))  # 复制图像并转换颜色空间
    img, info_img = preprocess(img, imgsize, jitter=0)  # 预处理图像
    img = np.transpose(img / 255., (2, 0, 1))  # 转置图像并归一化
    img = torch.from_numpy(img).float().unsqueeze(0)  # 将图像转换为张量

    if args.gpu >= 0:  # 如果指定了GPU
        model.cuda(args.gpu)  # 将模型移动到GPU
        img = Variable(img.type(torch.cuda.FloatTensor))  # 将图像转换为CUDA张量
    else:
        img = Variable(img.type(torch.FloatTensor))  # 将图像转换为张量

    assert args.weights_path or args.ckpt, 'One of --weights_path and --ckpt must be specified'  # 确保指定了权重文件或检查点文件

    if args.weights_path:  # 如果指定了权重文件
        print("loading yolo weights %s" % (args.weights_path))
        parse_yolo_weights(model, args.weights_path)  # 加载YOLO权重
    elif args.ckpt:  # 如果指定了检查点文件
        print("loading checkpoint %s" % (args.ckpt))
        state = torch.load(args.ckpt)  # 加载检查点
        if 'model_state_dict' in state.keys():  # 如果检查点包含模型状态字典
            model.load_state_dict(state['model_state_dict'])  # 加载模型状态字典
        else:
            model.load_state_dict(state)  # 加载模型状态

    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():  # 禁用梯度计算
        outputs = model(img)  # 对图像进行推理
        outputs = postprocess(outputs, 80, confthre, nmsthre)  # 对输出进行后处理

    if outputs[0] is None:  # 如果没有检测到对象
        print("No Objects Deteted!!")
        return

    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()  # 获取COCO标签名称

    bboxes = list()  # 创建一个空的边界框列表
    classes = list()  # 创建一个空的类别列表
    colors = list()  # 创建一个空的颜色列表

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:  # 对每个输出进行循环

        cls_id = coco_class_ids[int(cls_pred)]  # 获取类别id
        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' %
              (coco_class_names[cls_id], cls_conf.item()))  # 打印标签和置信度
        box = yolobox2label([y1, x1, y2, x2], info_img)  # 将YOLO框转换为标签
        bboxes.append(box)  # 将边界框添加到列表
        classes.append(cls_id)  # 将类别添加到列表
        colors.append(coco_class_colors[int(cls_pred)])  # 将颜色添加到列表

    if args.background:  # 如果在后台模式下运行
        import matplotlib
        matplotlib.use('Agg')  # 使用无显示的后端

    from utils.vis_bbox import vis_bbox
    import matplotlib.pyplot as plt

    vis_bbox(
        img_raw, bboxes, label=classes, label_names=coco_class_names,
        instance_colors=colors, linewidth=2)  # 可视化边界框
    plt.show()  # 显示图像

    if args.background:  # 如果在后台模式下运行
        plt.savefig('output.png')  # 保存图像


if __name__ == '__main__':
    main()  # 运行主函数
