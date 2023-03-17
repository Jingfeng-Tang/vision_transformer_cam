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

def multilabel_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


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
    # args = get_args()
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

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        names, images, labels = data
        sample_num += images.shape[0]
        pred, cams = model(images.to(device))
        if epoch > 495:
            generate_origin_cam(cams, labels, names)
        # a = torch.zeros(1)
        # b = a[10]
        # generate origin cams
        labels_sum = labels.sum(dim=1)  # 每个图中有几个类别
        pred_multihot = torch.zeros(pred.shape)  # pred all zero tensor
        for i in range(labels_sum.shape[0]):
            val, index = torch.topk(pred, labels_sum[i].int(), dim=1)
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

        data_loader.desc = "[train epoch {}] loss: {:.3f}, f1_score: {:.3f}".format(epoch,
                                                                                    accu_loss.item() / (step + 1),
                                                                                    f1_score)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), f1_score


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
