import os
import sys
import datetime
import random
import argparse
import torch
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model
from voc12.data import VOC12ClsDataset
import numpy as np
from tqdm import tqdm
from utils import compute_mAP, ConfusionMatrix, cam_norm
from torchvision.transforms import functional as F
import json
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt
torch.set_printoptions(threshold=np.inf)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seg_resize(seg_label):
    seg_label = F.resize(seg_label, 224, interpolation=transforms.InterpolationMode.NEAREST)

    return seg_label


def ToTensor(seg_label):
    target = torch.as_tensor(np.array(seg_label), dtype=torch.int64)

    return target


# 载入调色板
def load_palette():
    palette_path = "./palette.json"
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    return pallette


def val(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建seg_res保存文件夹
    seg_res_path = './validate_seg_pred/'
    if os.path.exists(seg_res_path) is False:
        os.makedirs(seg_res_path)

    # 创建validate_cam保存文件夹
    validate_cam_path = './validate_cam/'
    if os.path.exists(validate_cam_path) is False:
        os.makedirs(validate_cam_path)

    # 用来保存验证过程中信息
    val_log_txt = "validating_log_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 加载原图
    oriImgFolderpath = args.dataset_path+'JPEGImages/'

    # 加载调色板
    pallette = load_palette()

    # 遵循SEAM数据增强操作  目前这样数据增强只是为了可视化，使得cam图map到原图的相应位置
    data_transform = {
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                     }

    # dataset
    val_dataset = VOC12ClsDataset(img_name_list_path=args.val_img_name_path,
                                  voc12_root=args.dataset_path,
                                  transform=data_transform["val"],
                                  seg_label_flag=True)

    batch_size = args.batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # 加载模型权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(args.weights)
        print(model.load_state_dict(weights_dict, strict=False))

    date = datetime.date.today()
    print((str(date)))
    # validate
    model.eval()
    model.is_train = False
    mAP = []
    confmat = ConfusionMatrix(20)
    data_loader = tqdm(val_loader, file=sys.stdout)
    tags = ["mAP_multiple_class_label", "mIOU"]
    with torch.no_grad():
        validate_IOU = []
        for step, data in enumerate(data_loader):
            name, image, target, seg_labels = data
            b, h, w = seg_labels.shape
            image, target, seg_labels = image.to(device), target.to(device), seg_labels.to(device)
            output, attn_w, attn_m, allbs_hw_p_ts, clsh1_weight_ori, ori_allbs_hw_p_ts = model(image)
            # print('\n')
            # print(name)
            compute_allbs_hw_p_ts = allbs_hw_p_ts

            predcls = torch.sigmoid(allbs_hw_p_ts)  # 16*20
            predcls[predcls >= 0.9] = 1
            predcls[predcls < 0.9] = 0
            zero_t = torch.full((1, 768), -10, device='cuda:0')
            batchsize = 1
            for l in range(batchsize):
                clsh1_weight = clsh1_weight_ori  # 20*768
                clsh1_weight = clsh1_weight.clone().detach()
                for k in range(20):
                    if predcls[l][k] == 0:  # 如果不是模型所预测的那个类
                        clsh1_weight[k] = zero_t  # 将所属权重置0
                cls_to_768 = torch.argmax(clsh1_weight, dim=0)  # 为768个特征赋予类别
                # STEP1: 将768个特征与16个patch建立联系（为16个patch赋予类别）
                curimg_ori_allbs_hw_p_ts = ori_allbs_hw_p_ts[l]
                zero_16_768 = torch.arange(21, 12309, 1, device='cuda:0').reshape(768, 16)
                # 每个特征哪个patch贡献大
                contriPatchindex = torch.argmax(curimg_ori_allbs_hw_p_ts, dim=0)
                # print(f'contriPatchindex: {contriPatchindex.shape}')  # 16*20
                for m in range(768):
                    zero_16_768[m][contriPatchindex[m]] = cls_to_768[m]
                # print(f'zero_16_768.shape: {zero_16_768}')
                patch_to_cls, indice = torch.mode(zero_16_768, dim=0)
                # print(f'当前图片的16个hwpatch的分类： {patch_to_cls}')
                # STEP2: patch_to_cls是16个高权重patch所分配的类
                # STEP3: 将block5的mask的obj index与16个patch进行相似度计算，然后softmax，argmax，分配类别
                # block5_obj_index是block5的前景，接下来计算他们与16patch的余弦相似度（先用最后block的x）
                # print(f'x.shape: {x.shape}')      1*197*768
                # print(f'block5_obj_index.shape: {block5_obj_index.shape}')  # 1*196    1前景 0背景
                # print(f'index.shape: {index}')  # 16  index序号

                # 所有patch的特征
                patchebed = attn_m[11].squeeze(0)[1:, :]  # 196*768
                # hw patch的特征
                hw_patch_ebed = ori_allbs_hw_p_ts.squeeze(0)  # 16*768

                # 方法二 将hwp与所有的patch进行相似度计算，确定目标区域 ，插值， argmax
                c_obj_ebed = torch.nn.functional.normalize(patchebed, dim=1)  # 196*768
                c_hw_ebed = torch.nn.functional.normalize(hw_patch_ebed, dim=1)  # 16*768
                seglabel_16 = []
                for hwpinx in range(16):
                    # print(f'c_hw_ebed[hwpinx]: {c_hw_ebed[hwpinx].shape}')
                    # print(f'c_obj_ebed.t(): {c_obj_ebed.t().shape}')
                    cos_sim = torch.einsum('ij,jk->ik', c_hw_ebed[hwpinx].unsqueeze(0), c_obj_ebed.t()).reshape(14, 14)
                    # print(f'cos_sim.shape: {cos_sim.shape}')
                    # 插值回原图大小
                    cos_ori_size = torch.nn.functional.interpolate(cos_sim.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
                    seglabel_16.append(cos_ori_size.squeeze(0).squeeze(0))
                seglabel_16_t = torch.stack(seglabel_16)
                final_seg = seglabel_16_t.argmax(dim=0)
                final_seg_v, inx = seglabel_16_t.max(dim=0)
                # print(type(final_seg_v))
                # final_seg_v 是余弦相似度值，如果某个通道是最大的，但是值很低，判定为背景类，生成mask
                mask_bg_cos_threshold = 0.5
                final_seg_v[final_seg_v < mask_bg_cos_threshold] = 0
                final_seg_v[final_seg_v >= mask_bg_cos_threshold] = 1

                # print(f'final_seg.shape: {final_seg}')
                # 加个偏移量 offset  防止变换后产生冲突
                offset_mat = torch.full((h, w), 50, device='cuda:0')
                final_seg = final_seg + offset_mat  # 都是50+的
                # 获取16pwp的类别映射
                final_seg[final_seg == 50] = patch_to_cls[0]+1
                final_seg[final_seg == 51] = patch_to_cls[1]+1
                final_seg[final_seg == 52] = patch_to_cls[2]+1
                final_seg[final_seg == 53] = patch_to_cls[3]+1
                final_seg[final_seg == 54] = patch_to_cls[4]+1
                final_seg[final_seg == 55] = patch_to_cls[5]+1
                final_seg[final_seg == 56] = patch_to_cls[6]+1
                final_seg[final_seg == 57] = patch_to_cls[7]+1
                final_seg[final_seg == 58] = patch_to_cls[8]+1
                final_seg[final_seg == 59] = patch_to_cls[9]+1
                final_seg[final_seg == 60] = patch_to_cls[10]+1
                final_seg[final_seg == 61] = patch_to_cls[11]+1
                final_seg[final_seg == 62] = patch_to_cls[12]+1
                final_seg[final_seg == 63] = patch_to_cls[13]+1
                final_seg[final_seg == 64] = patch_to_cls[14]+1
                final_seg[final_seg == 65] = patch_to_cls[15]+1
                # 已经获取到final_seg了，还需要背景mask--------------------------------------------------------------------

                # # 取第五个block权重来分割前景背景--------------------------------------------------------------------------
                # # weights_i是第五个block的权重
                # # print(f'attn_w[4].shape: {attn_w[4].shape}')      # 1*12*197*197
                # att_mat = torch.mean(attn_w[4], dim=1)  # 1 * 197 * 197: batchsize * patches * patches
                # # print(f'att_mat.shape: {att_mat.shape}')
                # # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
                # residual_att = torch.eye(att_mat.size(2)).cuda()  # 197 * 197 identity matrix
                # aug_att_mat = att_mat + residual_att
                # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # 16 * 197 * 197
                # mask_i = aug_att_mat[:, 0, 1:]  # 1*196
                # mask_14 = mask_i / mask_i.max()  # 1*196
                # # 取第五个block权重来分割前景背景--------------------------------------------------------------------------

                # 取第6-12个block权重来分割前景背景--------------------------------------------------------------------------
                att_mat = torch.stack(attn_w).squeeze(1)        # 12*12*197*197  block*head*
                # 取6-12个blocks
                att_mat = att_mat[5:, :, :, :]      # 7*12*197*197
                att_mat = torch.mean(att_mat, dim=0)    # 12*197*197
                att_mat = torch.mean(att_mat, dim=0).unsqueeze(0)  # 1 * 197 * 197: batchsize * patches * patches
                # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
                residual_att = torch.eye(att_mat.size(2)).cuda()  # 197 * 197 identity matrix
                aug_att_mat = att_mat + residual_att
                aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # 16 * 197 * 197
                mask_i = aug_att_mat[:, 0, 1:]  # 1*196
                mask_14 = mask_i / mask_i.max()  # 1*196
                # 取第6-12个block权重来分割前景背景--------------------------------------------------------------------------

                # 插值回原图大小，然后设置阈值
                bg_ori_size = torch.nn.functional.interpolate(mask_14.reshape(14, 14).unsqueeze(0).unsqueeze(0),
                                                              size=(h, w), mode='bilinear', align_corners=False)
                bg_ori_size = bg_ori_size.squeeze(0).squeeze(0)
                # print(f'bg_ori_size.shape: {bg_ori_size.shape}')
                # 获取小于0.25的权重的索引
                bg_threshold = 0.05
                bg_ori_size[bg_ori_size < bg_threshold] = 0
                bg_ori_size[bg_ori_size >= bg_threshold] = 1

                mask_final_fg = torch.einsum('ij,ij->ij', final_seg_v, bg_ori_size)

                # # 显示背景与前景的分割
                toimg = transforms.ToPILImage()
                # seg_fb = torch.as_tensor(bg_ori_size, dtype=torch.uint8)
                # seg_fb_save = toimg(seg_fb)
                # seg_fb_save.putpalette(pallette)
                # seg_fb_save.save("./validate_seg_pred/"+name[0]+'___bg'+".png")

                final_seg_res = torch.einsum('ij,ij->ij', final_seg, mask_final_fg)
                seg_obj = torch.as_tensor(final_seg_res, dtype=torch.uint8)

                seg_obj_save = toimg(seg_obj)
                seg_obj_save.putpalette(pallette)
                # seg_obj_save.save("./final_seg/"+name[0]+".png")


            # output = torch.sigmoid(output)      # class tokens用于多标签分类
            output = torch.sigmoid(compute_allbs_hw_p_ts)      # 16hwp用于多标签分类


            # 计算mAP
            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            mean_ap = np.mean(mAP_list)
            mean_ap_all = np.mean(mAP)

            # 计算mIOU
            confmat.update(seg_labels.flatten(), seg_obj.flatten())

            seg_obj_save.save("./validate_seg_pred/" + name[0] + ".png")

            data_loader.desc = "[test step {}] cur_step_mAP: {:.3f} all_step_mAP: {:.3f}"\
                .format(step,
                        mean_ap,
                        mean_ap_all,
                        )
        print(confmat)

    # write into txt
    with open(val_log_txt, "a") as f:
        # 记录每个step对应的测试过程中的各指标
        val_log = f"[step: {step}]\n" \
                   f"mAP_multiple_class_label: {mean_ap:.5f}     "
        f.write(val_log + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('--model_name', type=str, default='vit_base', required=True, help='create model name')
    parser.add_argument('--num_classes', type=int, default=20, required=True)
    parser.add_argument('--weights', type=str, default='/data/c425/tjf/vit/weights_head1/2023-04-05-cur_ep78-bestloss.pth',
                        required=False,
                        help='initial weights path, set to null character if you do not want to load weights')
    # 验证参数
    parser.add_argument('--batch_size', type=int, default=64, required=True)
    # 文件路径
    parser.add_argument('--dataset_path', type=str,
                        default="/data/c425/tjf/datasets/VOC2012/", required=True)
    parser.add_argument('--val_img_name_path', type=str,
                        default="/data/c425/tjf/vit/voc12/val.txt", required=True)
    parser.add_argument('--ori_cam_path', type=str,
                        default="/data/c425/tjf/vit/origincams/", required=True)
    # 设备参数
    parser.add_argument('--device', default='cuda:0', type=str, required=True, help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    same_seeds(0)   # 随机化种子

    # a = torch.tensor(([1, 2, 5], [6, 7, 8], [12, 65, 89]))
    # print(a.shape)
    #
    # b = a[:, 0:2]
    # print(b)
    # a = []
    # b = a[10]

    val(opt)
