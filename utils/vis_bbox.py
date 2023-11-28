import numpy as np
import matplotlib.pyplot as plt


def vis_bbox(img, bbox, label=None,score=None,label_names=None,
             instance_colors=None,alpha=1, linewidth=3.,ax=None):
    """_summary_

    Args:
        img (~numpy.ndarray):形状为 :math:(3, height, width) 的数组。这是RGB格式，其值的范围是 :math:[0, 255]。如果此项为 None，则不显示任何图像。
        bbox (_type_):形状为 :math:(R, 4) 的数组，其中 :math:R 是图像中边界框的数量。第二轴中的每个元素由 :math:(y_{min}, x_{min}, y_{max}, x_{max}) 组成
        label (_type_, optional):(~numpy.ndarray): 形状为 :math:(R,) 的整数数组。值对应于存储在 label_names 中的标签名称的id。这是可选的
        label_names (_type_, optional): _description_. Defaults to None.
        instance_colors (_type_, optional): _description_. Defaults to None.
        alpha (int, optional): _description_. Defaults to 1.
        linewidth (_type_, optional): _description_. Defaults to 3..
        ax (_type_, optional): _description_. Defaults to None.
    """
    
    # input check
    if label is not None and not len(bbox) == len(label):
        raise ValueError("Lenth of the bbox must as same as label")
    if score is not None and not len(bbox) == len(score):
        raise ValueError("lenth of the score must as same as bbox")
    
    # 判断ax画布是否被创建
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    ax.imshow(img.transpose((1, 2, 0)).astype(np.uint8))
    #(3, height, width)->(height, width, 3)
    
    if len(bbox) == 0:
        return ax
    #处理边框颜色
    if instance_colors is None:
        instance_colors = np.zeros((len(bbox),3),dtype=np.float32)
        instance_colors[:,0] = 255
    instance_colors = np.array(instance_colors)
    
    for i,bb in enumerate(bbox):
        xy = (bb[1],bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        color = instance_colors(i % len(instance_colors)) /255
        ax.add_patch(plt.Rectangle(
            xy,width,height,fill=False,
            edgecolor=color,linewidth = linewidth,alpha=alpha))
        caption = []
        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb< len(label_names)):
                raise ValueError("No correspoding name is given")
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))
        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor':'white','alpha':0.7,'pad':10})
    return ax
