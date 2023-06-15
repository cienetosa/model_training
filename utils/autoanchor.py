
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm
import pandas as pd

from utils.general import LOGGER, colorstr, emojis

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """
    Check anchor fit to data, recompute if necessary
    yolov5 中不是只使用默认锚定框，在开始训练之前会对数据集中标注信息进行核查，计算此数据集标注信息针对默认锚定框的最佳召回率，
    当最佳召回率大于或等于0.98，则不需要更新锚定框；如果最佳召回率小于0.98，则需要重新计算符合此数据集的锚定框。
    """
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh
    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold 是大于0小于9的数
        bpr = (best > 1 / thr).float().mean()  # best possible recall [1,1,1...0,0,...] 该矩阵求均值，是0-1之间的小数
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides 8,16,32
    anchors = m.anchors.clone() * stride  # current anchors  yolov5s.yaml中的默认'anchors'
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(emojis(s))


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]  # 70000*9*2 共有70000个真实框，每个真实框都除以n(9)个自动锚框
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric  70000*9 每一行的9个元素表示该真实框对于锚框的最大伸缩比例 
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x  在每个伸缩比例中挑选出一个伸缩最小的锚框 70000*1

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        # best > thr).float() 满足大于..条件的返回1，否则返回0.
        # best * (best > thr).float() 满足大于..条件的返回best中对应的原值，否则返回0.        
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large 将n个聚类中心(宽高)的两值相乘，得出每个聚类锚框的面积，然后按从小到大的顺序排列
        x, best = metric(k, wh0)
        # 这个伸缩比例小于15倍或大于1/15倍，可判断为best。bpr表示在与真实框的匹配中，判断为best的锚框占所有锚框的比例。
        # aat表示在所有真实框中，每个真实框平均与几个锚框的匹配是合理的。        
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # dataset.labels:70000*5,每一行中有类别id，box的四个坐标。shapes:每张被resize(最大边640)后的图片长高。
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 将原来box的长高坐标映射成resize后的大小
    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found: {i} of {len(wh0)} labels are < 3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels 这里面过滤了高<2的标注框，少了7000个框！
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1
    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening  对每一列求标准差  1*2
        k = kmeans(wh / s, n, iter=30)[0] * s  # points 点乘，得出n个聚类中心 n*2
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING: switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False) # 将n个聚类中心(宽高)的两值相乘，得出每个聚类锚框的面积，然后按从小到大的顺序排列

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            # 对v矩阵[9,2]进行变异来对聚类出来的k个中心进行些许变化，其中v的数据在1左右轻微浮动
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        # 重新计算适应度，如果更好则进化
        fg = anchor_fitness(kg)
        if fg > f:  # 如果变异后的适用度比上一轮的好，那么就进化，这里是用宽高比的平均值来表述适应度的选择
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)  # 按面积排序，分配给3个特征层

    return print_results(k)
