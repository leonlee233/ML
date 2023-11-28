from __future__ import division

from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *

import os
import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim

def parse_args():
    """
    解析命令行参数。
    Returns:
        args : 包含所有命令行参数的命名空间。
    """
    parser = argparse.ArgumentParser()  # 创建一个解析器对象
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',
                        help='配置文件。参见readme')  # 添加一个命令行参数，指定配置文件的路径
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet权重文件')  # 添加一个命令行参数，指定权重文件的路径
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='工作线程的数量')  # 添加一个命令行参数，指定工作线程的数量
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='保存检查点的间隔')  # 添加一个命令行参数，指定保存检查点的间隔
    parser.add_argument('--eval_interval', type=int,
                            default=4000, help='评估间隔')  # 添加一个命令行参数，指定评估的间隔
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch检查点文件路径')  # 添加一个命令行参数，指定检查点文件的路径
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='保存检查点文件的目录')  # 添加一个命令行参数，指定保存检查点文件的目录
    parser.add_argument('--use_cuda', type=bool, default=True)  # 添加一个命令行参数，指定是否使用CUDA
    parser.add_argument('--debug', action='store_true', default=False,
                        help='调试模式，只训练一张图像')  # 添加一个命令行参数，指定是否启用调试模式
    parser.add_argument(
        '--tfboard', help='用于记录的tensorboard路径', type=str, default=None)  # 添加一个命令行参数，指定tensorboard的路径
    return parser.parse_args()  # 解析命令行参数并返回



def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()  # 解析命令行参数
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda  # 检查是否可以使用CUDA
    os.makedirs(args.checkpoint_dir, exist_ok=True)  # 创建检查点目录

    # 解析配置设置
    with open(args.cfg, 'r') as f:  # 打开配置文件
        cfg = yaml.load(f)  # 加载配置文件

    print("successfully loaded config file: ", cfg)

    # 从配置文件中获取训练参数
    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # 学习率设置
    def burnin_schedule(i):
        if i < burn_in:
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # 初始化模型
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    if args.weights_path:  # 如果指定了权重文件路径
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)  # 加载YOLO权重
    elif args.checkpoint:  # 如果指定了检查点文件路径
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)  # 加载检查点
        if 'model_state_dict' in state.keys():  # 如果检查点包含模型状态字典
            model.load_state_dict(state['model_state_dict'])  # 加载模型状态字典
        else:
            model.load_state_dict(state)  # 加载模型状态

    if cuda:  # 如果可以使用CUDA
        print("using cuda") 
        model = model.cuda()  # 将模型移动到GPU

    if args.tfboard:  # 如果使用TensorBoard
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard)  # 创建一个TensorBoard记录器

    model.train()  # 将模型设置为训练模式

    imgsize = cfg['TRAIN']['IMGSIZE']  # 获取图像大小
    dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
                  data_dir='COCO/',
                  img_size=imgsize,
                  augmentation=cfg['AUGMENTATION'],
                  debug=args.debug)  # 创建一个COCO数据集

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)  # 创建一个数据加载器
    dataiterator = iter(dataloader)  # 创建一个数据迭代器

    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir='COCO/',
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'])  # 创建一个COCO API评估器

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # 设置数据类型

    # 设置优化器
    # 仅在 conv.weight 上设置权重衰减
    params_dict = dict(model.named_parameters())  # 获取模型的所有参数
    params = []
    for key, value in params_dict.items():  # 遍历所有参数
        if 'conv.weight' in key:  # 如果参数是卷积层的权重
            params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]  # 添加到参数列表中，并设置权重衰减
        else:
            params += [{'params':value, 'weight_decay':0.0}]  # 否则，添加到参数列表中，但不设置权重衰减
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)  # 创建一个SGD优化器
    
    iter_state = 0  # 初始化迭代状态
    
    if args.checkpoint:  # 如果提供了检查点文件
        if 'optimizer_state_dict' in state.keys():  # 如果状态字典中有优化器的状态
            optimizer.load_state_dict(state['optimizer_state_dict'])  # 加载优化器的状态
            iter_state = state['iter'] + 1  # 更新迭代状态
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)  # 创建一个LambdaLR调度器
    
    # 开始训练循环
    for iter_i in range(iter_state, iter_size + 1):  # 对每一个迭代
    
        # COCO评估
        if iter_i % args.eval_interval == 0 and iter_i > 0:  # 如果到达评估间隔
            ap50_95, ap50 = evaluator.evaluate(model)  # 进行评估
            model.train()  # 将模型设置为训练模式
            if args.tfboard:  # 如果使用tensorboard
                tblogger.add_scalar('val/COCOAP50', ap50, iter_i)  # 记录AP50
                tblogger.add_scalar('val/COCOAP50_95', ap50_95, iter_i)  # 记录AP50_95
    
        # 子划分循环
        optimizer.zero_grad()  # 清零梯度
        for inner_iter_i in range(subdivision):  # 对每一个子划分
            try:
                imgs, targets, _, _ = next(dataiterator)  # 加载一个批次
            except StopIteration:  # 如果数据迭代器耗尽
                dataiterator = iter(dataloader)  # 重新创建数据迭代器
                imgs, targets, _, _ = next(dataiterator)  # 加载一个批次
            imgs = Variable(imgs.type(dtype))  # 将图像转换为Variable
            targets = Variable(targets.type(dtype), requires_grad=False)  # 将目标转换为Variable
            loss = model(imgs, targets)  # 计算损失
            loss.backward()  # 反向传播
    
        optimizer.step()  # 更新参数
        scheduler.step()  # 更新学习率
    
        if iter_i % 10 == 0:  # 如果到达记录间隔
            # 记录
            current_lr = scheduler.get_lr()[0] * batch_size * subdivision  # 计算当前学习率
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, current_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'], 
                     model.loss_dict['l2'], imgsize),
                  flush=True)  # 打印训练信息
    
            if args.tfboard:  # 如果使用tensorboard
                tblogger.add_scalar('train/total_loss', model.loss_dict['l2'], iter_i)  # 记录总损失
    
            # 随机调整大小
            if random_resize:  # 如果启用了随机调整大小
                imgsize = (random.randint(0, 9) % 10 + 10) * 32  # 随机生成一个新的图像大小
                dataset.img_shape = (imgsize, imgsize)  # 更新数据集的图像形状
                dataset.img_size = imgsize  # 更新数据集的图像大小
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)  # 创建一个新的数据加载器
                dataiterator = iter(dataloader)  # 创建一个新的数据迭代器
    
        # 保存检查点
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):  # 如果迭代次数大于0且是检查点间隔的倍数
            torch.save({'iter': iter_i,  # 保存当前的迭代次数
                        'model_state_dict': model.state_dict(),  # 保存模型的状态字典
                        'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的状态字典
                        },
                       os.path.join(args.checkpoint_dir, "snapshot"+str(iter_i)+".ckpt"))  # 将检查点保存到指定的目录

    if args.tfboard:  # 如果使用TensorBoard
        tblogger.close()  # 关闭TensorBoard记录器


if __name__ == '__main__':
    main()  # 运行主函数
