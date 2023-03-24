import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import cv2
from sklearn.metrics import average_precision_score


def multilabel_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes+1
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)+1)        # +1 防止 nan

        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

    def get_mIOU(self):
        acc_global, acc, iu = self.compute()
        # if iu.
        return iu.mean().item() * 100


def cam_norm(cam):
    cam = cam.detach().cpu()
    cam = np.array(cam)
    # 归一化
    cam = cam - np.min(cam)  # 减去最小，除以最大，目的是归一化
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)

    return cam_img


def generate_origin_cam(cams, labels, names):
    # print(names[0])
    # print(f'cams.shape: {cams.shape}')
    # print(f'labels.shape: {labels.shape}')
    cams = cams.permute(0, 2, 1).reshape(32, 20, 14, 14)        # labels 32*20
    cam_savepth = '/data/c425/tjf/vit/origincams/'
    # print(f'cams.shape: {cams.shape}')
    # 按照labels取cams
    for i in range(32):
        val, index = torch.topk(labels[i], int(labels[i].sum().item()), dim=0)
        # 每张图
        # base_cam = torch.zeros(14, 14)
        cam_percls = []
        img_path = '/data/c425/tjf/datasets/VOC2012/JPEGImages/'
        img_name = names[i] + '.jpg'
        for j in range(index.shape[0]):
            # 按 int(index[j])
            cam_percls.append(cams[i][int(index[j])])
            # 存一下每张图的label类别激活映射图
            cam_norm_i = cam_norm(cams[i][int(index[j])])
            img_i = cv2.imread(img_path + img_name)
            height, width, _ = img_i.shape
            heatmap = cv2.applyColorMap(cv2.resize(cam_norm_i, (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img_i * 0.5
            siglabel_cam_name = names[i] + '_siglabel_cam.jpg'
            cv2.imwrite(cam_savepth+siglabel_cam_name, result)

        base_cam = torch.stack(cam_percls, dim=0)
        # print(base_cam)
        label_cls_cam, _ = base_cam.max(dim=0)
        # print('----------------')
        # print(label_cls_cam.shape)
        cam_normed = cam_norm(label_cls_cam)
        img = cv2.imread(img_path+img_name)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam_normed, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        syn_cam_name = names[i]+'_syn_cam.jpg'
        cv2.imwrite(cam_savepth+syn_cam_name, result)


def generate_pseudo_result(cam, oriimg_path):
    img = cv2.imread(oriimg_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
    # 设置阈值，大于阈值（类别区域），小于阈值（背景区域）
    threshold = 200
    # 小于阈值，置0
    # heatmap 什么类型？

    # 加个背景 5e-5 大于0的极小值


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    local_rank = torch.distributed.get_rank()
    # print(local_rank)
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        names, images, labels = data
        sample_num += images.shape[0]
        pred, cams, attn_w, attn_m = model(images.to(device, non_blocking=True))
        # pred = torch.nn.functional.softmax(pred, dim=1)
        # pred = torch.sigmoid(pred)
        # if epoch > 495:
        #     generate_origin_cam(cams, labels, names)
        # generate origin cams
        labels_sum = labels.sum(dim=1)  # 每个图中有几个类别
        pred_multihot = torch.zeros(pred.shape)  # pred all zero tensor
        for i in range(labels_sum.shape[0]):
            # ？？？？？？？？？？？？？？？？？
            val, index = torch.topk(pred, labels_sum[i].int(), dim=1)       # 注意这里是不是不需要softmax？？？？？？
            for j in range(labels_sum[i].int()):
                pred_multihot[i][index[i][j]] = 1
        pred_multihot_np = pred_multihot.numpy()
        labels_np = labels.numpy()
        f1_score_i = 0.0
        for k in range(labels_sum.shape[0]):
            f1_score_i = multilabel_score(labels_np[k], pred_multihot_np[k])
            f1_score_i += f1_score_i
        f1_score = f1_score_i/labels_sum.shape[0]

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        # data_loader.desc = "[train epoch {}] loss: {:.3f}, f1_score: {:.3f}".format(epoch,
        #                                                                             accu_loss.item() / (step + 1),
        #                                                                             f1_score)
        if local_rank == 1:
            data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch,
                                                                  accu_loss.item() / (step + 1),
                                                                  )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), f1_score


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes):
    model.eval()
    local_rank = torch.distributed.get_rank()
    mAP = []
    # confmat = ConfusionMatrix(num_classes)
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            name, image, target, seg_labels = data

            image, target = image.to(device), target.to(device)

            output, cams, attn_w, attn_m = model(image)

            output = torch.sigmoid(output)

            # 计算mAP
            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            mean_ap = np.mean(mAP_list)
            mean_ap_all = np.mean(mAP)
            # 计算mIOU
            if local_rank == 1:
                data_loader.desc = "[test epoch {}] cur_step_mAP: {:.3f} all_step_mAP: {:.3f}".format(epoch,
                                                                                                      mean_ap,
                                                                                                      mean_ap_all)

    return mean_ap_all


def compute_mAP(labels, outputs):

    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    # print(f'y_true.shape[0]: {y_true.shape[0]}')
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            # print(f'y_true[i]: {y_true[i]}')
            # print(f'y_pred[i]: {y_pred[i]}')
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)

    return AP


if __name__ == '__main__':
    labels = torch.tensor([1, 0, 1, 0, 0, 0])
    print(labels)
    outputs = torch.tensor([0.98, 0.3, 0.86, 0.85, 0.36, 0.48])
    ap = average_precision_score(labels, outputs)       # ok

    print(ap)