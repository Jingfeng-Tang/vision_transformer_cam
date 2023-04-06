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
    seg_res_path = './validate_seg_res/'
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
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # 加载模型权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
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
            output, cams, attn_w, attn_m, allbs_hw_p_ts, clsh1_weight_ori, ori_allbs_hw_p_ts = model(image)
            print('\n')
            print(name)

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
                # print(f'clsh1_weight.shape: {clsh1_weight.shape}')        20*768
                # clsh1_softmax = torch.softmax(clsh1_weight, dim=0)
                # print(clsh1_softmax)
                cls_to_768 = torch.argmax(clsh1_weight, dim=0)  # 为768个特征赋予类别
                # STEP1: 将768个特征与16个patch建立联系（为16个patch赋予类别）
                curimg_ori_allbs_hw_p_ts = ori_allbs_hw_p_ts[l]
                zero_16_768 = torch.arange(21, 12309, 1, device='cuda:0').reshape(768, 16)
                # print(f'zero_16_768: {zero_16_768}')
                # 每个特征哪个patch贡献大
                contriPatchindex = torch.argmax(curimg_ori_allbs_hw_p_ts, dim=0)
                # print(f'contriPatchindex: {contriPatchindex.shape}')  # 16*20
                # print(f'contriPatchindex: {contriPatchindex}')
                for m in range(768):
                    zero_16_768[m][contriPatchindex[m]] = cls_to_768[m]
                # print(f'zero_16_768.shape: {zero_16_768}')
                patch_to_cls, indice = torch.mode(zero_16_768, dim=0)
                print(f'当前图片的16个hwpatch的分类： {patch_to_cls}')
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
                # print(f'final_seg.shape: {final_seg}')
                # 加个偏移量 offset  防止变换后产生冲突
                offset_mat = torch.full((h, w), 50, device='cuda:0')
                final_seg = final_seg + offset_mat  # 都是50+的
                # 获取16pwp的类别映射
                final_seg[final_seg == 50] = patch_to_cls[0]
                final_seg[final_seg == 51] = patch_to_cls[1]
                final_seg[final_seg == 52] = patch_to_cls[2]
                final_seg[final_seg == 53] = patch_to_cls[3]
                final_seg[final_seg == 54] = patch_to_cls[4]
                final_seg[final_seg == 55] = patch_to_cls[5]
                final_seg[final_seg == 56] = patch_to_cls[6]
                final_seg[final_seg == 57] = patch_to_cls[7]
                final_seg[final_seg == 58] = patch_to_cls[8]
                final_seg[final_seg == 59] = patch_to_cls[9]
                final_seg[final_seg == 60] = patch_to_cls[10]
                final_seg[final_seg == 61] = patch_to_cls[11]
                final_seg[final_seg == 62] = patch_to_cls[12]
                final_seg[final_seg == 63] = patch_to_cls[13]
                final_seg[final_seg == 64] = patch_to_cls[14]
                final_seg[final_seg == 65] = patch_to_cls[15]
                # print(f'final_seg.shape: {final_seg}')
                # 已经获取到final_seg了，还需要背景mask
                # print(f'attn_w[4].shape: {attn_w[4].shape}')
                # weights_i是第五个block的权重
                att_mat = torch.mean(attn_w[4], dim=1)  # 1 * 197 * 197: batchsize * patches * patches
                # print(f'att_mat.shape: {att_mat.shape}')
                # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
                residual_att = torch.eye(att_mat.size(2)).cuda()  # 197 * 197 identity matrix
                aug_att_mat = att_mat + residual_att
                aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # 16 * 197 * 197
                mask_i = aug_att_mat[:, 0, 1:]  # 1*196
                mask_14 = mask_i / mask_i.max()  # 1*196
                # 插值回原图大小，然后设置阈值
                bg_ori_size = torch.nn.functional.interpolate(mask_14.reshape(14, 14).unsqueeze(0).unsqueeze(0),
                                                              size=(h, w), mode='bilinear', align_corners=False)
                bg_ori_size = bg_ori_size.squeeze(0).squeeze(0)
                print(f'bg_ori_size.shape: {bg_ori_size.shape}')
                # 获取小于0.25的权重的索引
                bg_ori_size[bg_ori_size < 0.25] = 0
                bg_ori_size[bg_ori_size >= 0.25] = 1

                # # 可视化每张图
                # show_seg = seg_lab.squeeze(0).squeeze(0).cpu().numpy()
                # plt.imshow(show_seg)
                # plt.show()

                # 方法一 只将hwp与objp进行相似度计算，确定前景
                # final_cls = torch.zeros(196, device='cuda:1')
                # for objpindex in range(block5_obj_index.shape[1]):      # 循环196个patch
                #     curp_cls = 0
                #     # print(f'----------当前第{objpindex}个patch')
                #     max_cos_sim = -2.0
                #     if block5_obj_index[0][objpindex] == 1:      # 找到目标patch
                #         # print(f'找到index：{objpindex} 为前景patch')
                #         for hwpindex in range(16):
                #             # 计算余弦相似度
                #             # print(f'将该前景patch与高权重patch的相似度做比较')
                #             c_obj_ebed = F.normalize(patchebed[objpindex].unsqueeze(0))
                #             c_hw_ebed = F.normalize(hw_patch_ebed[hwpindex].unsqueeze(0))
                #             cur_cos = c_obj_ebed.mm(c_hw_ebed.t())
                #             # print(f'cur_cos: {cur_cos}')
                #             if cur_cos > max_cos_sim:
                #                 # print(f'找到')
                #                 curp_cls = patch_to_cls[hwpindex]+1
                #                 # print(f'当前patch被分为：{curp_cls} 类')
                #                 # 为当前目标patch所找到类 curp_cls
                #     # print(f'final_cls.shape: {final_cls.shape}')
                #     final_cls[objpindex] = curp_cls       # +1为了修正调色板索引
                #     # print(f'当前的全图patch分类情况：{final_cls} ')
                #
                # # # ---------------------------------可视化目标patch
                # #
                # # final_cls = block5_obj_index.reshape(14, 14)
                # #
                # # seg_rres = torch.as_tensor(final_cls, dtype=torch.uint8)
                # # toimg = transforms.ToPILImage()
                # # mask = toimg(seg_rres)
                # # mask.putpalette(pallette)
                # # self.final_seg_count = self.final_seg_count+1
                # # final_seg_count_str = str(self.final_seg_count)
                # # mask.save("./final_seg/"+final_seg_count_str+".png")
                # #
                # # # ---------------------------------可视化目标patch
                #
                #
                # final_cls = final_cls.reshape(14, 14)
                # # print(f'reshape后当前的全图patch分类情况：{final_cls} ')
                #
                # seg_rres = torch.as_tensor(final_cls, dtype=torch.uint8)
                # # print(f'seg_rres：{seg_rres} ')
                # toimg = transforms.ToPILImage()
                # mask = toimg(seg_rres)
                # mask.putpalette(pallette)
                # self.final_seg_count = self.final_seg_count+1
                # final_seg_count_str = str(self.final_seg_count)
                # mask.save("./final_seg/"+final_seg_count_str+".png")

            # allbs_hw_p_ts = allbs_hw_p_ts.reshape(batchsize, 4, 4, 768).permute(0, 3, 1, 2)     #  16 * 768 *4 *4

            # allbs_hw_p_ts = self.patch_d1(allbs_hw_p_ts)    # 16*256*4*4
            # allbs_hw_p_ts = allbs_hw_p_ts.permute(0, 2, 3, 1)       # 16 * 4*4*256
            # allbs_hw_p_ts = self.norm1(allbs_hw_p_ts)
            # allbs_hw_p_ts = allbs_hw_p_ts.permute(0, 3, 1, 2)
            # allbs_hw_p_ts = self.relu(allbs_hw_p_ts)
            # allbs_hw_p_ts = self.patch_d2(allbs_hw_p_ts)
            # allbs_hw_p_ts = allbs_hw_p_ts.permute(0, 2, 3, 1)  # 16 * 4*4*256
            # allbs_hw_p_ts = self.norm2(allbs_hw_p_ts)
            # allbs_hw_p_ts = allbs_hw_p_ts.permute(0, 3, 1, 2)
            # allbs_hw_p_ts = self.relu(allbs_hw_p_ts)
            # allbs_hw_p_ts = self.patch_d3(allbs_hw_p_ts)
            # allbs_hw_p_ts = self.relu(allbs_hw_p_ts)
            #
            # allbs_hw_p_ts = allbs_hw_p_ts.reshape(batchsize, 16)
            #
            # allbs_hw_p_ts = self.hwp_map_labels(allbs_hw_p_ts)

            # if len(self.blocks) - i <= 12:
            #     attn_weights.append(weights_i)
            #     attn_matrix.append(x)


            # cam-------------------------------------------------------------------------------------------------------
            # cams_hw = cams.reshape(b, 14, 14, 20).permute(0, 3, 1, 2)
            # cam_show = cams_hw
            #
            # print(cams_hw[0][0])
            # 先sigmoid试一试
            # cams_hw = torch.sigmoid(cams_hw)
            # 归一化
            # for i in range(cams_hw.shape[0]):
            #     for j in range(cams_hw.shape[1]):
            #         cams_hw[i][j] = cams_hw[i][j] - torch.min(cams_hw[i][j])
            #         cams_hw[i][j] = cams_hw[i][j] / torch.max(cams_hw[i][j])
            # cam-------------------------------------------------------------------------------------------------------

            output = torch.sigmoid(output)      # class tokens用于多标签分类
            max_pred_cls = output.argmax(dim=1).cpu()
            # 计算mAP
            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            mean_ap = np.mean(mAP_list)
            mean_ap_all = np.mean(mAP)
            # 计算mIOU
            # 生成attention map 12个block融合
            # attention map--------------------------------------------------------------------------------------------------
            # first, you should return all attention weights  in self-attention model (12 stages), and then stack them.
            att_mat = torch.stack(attn_w).squeeze(1)  # 12 * 12 * 197 * 768: block * heads * patches * embeddings
            # 对每一个头做平均（多头其实就是多个卷积核），那就类似于1*1卷积操作
            att_mat = torch.mean(att_mat, dim=1)  # 12 * 197 * 768: block * patches * embeddings
            # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(1))  # 768 * 768 初等矩阵
            att_mat = att_mat.cpu()
            aug_att_mat = att_mat + residual_att    # 12 * 197 * 197
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)   # 12 * 197 * 197
            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size())  # 12 * 197 * 197 零矩阵
            joint_attentions[0] = aug_att_mat[0]    # 联合注意力矩阵第一个维度=vit第一个block的注意力权重矩阵
            for n in range(1, aug_att_mat.size(0)):         # 1-11
                # joint第一个block = att第一个block * joint第零个block  ，就是迭代地累乘权重
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
            # Attention from the output token to the input space.
            v = joint_attentions[-1]        # 取累乘12次的权重矩阵   197 * 197
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))      # 14
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), (w, h))    # [..., np.newaxis]

            # a = []
            # b = a[10]
            # result = (mask * img).astype("uint8")     # 可视化
            # attention map---------------------------------------------------------------------------------------------

            # cam-------------------------------------------------------------------------------------------------------
            # # 将20个类别的14*14cam插值回原图大小
            # cam_label = torch.nn.functional.interpolate(cams_hw, size=(h, w), mode='bilinear', align_corners=False)
            # # 设置阈值，将每个类的最大激活区域找出来，其他区域置0
            # high_threshold = 0.916
            # # print(cam_label[0][0])
            # cam_label[cam_label < high_threshold] = 0
            # # 添加背景类
            # bg_label = torch.full((b, 1, h, w), 1e-5, device=device)
            # # 整合背景类与目标类
            # cam_bg_obj = torch.cat((bg_label, cam_label), dim=1)
            # # cam生成的分割预测
            # cam_segPred = torch.argmax(cam_bg_obj, dim=1)
            # # 计算mIOU
            # confmat.update(seg_labels.flatten(), cam_segPred.flatten())
            # cur_step_mean_IOU = confmat.get_mIOU()
            # validate_IOU.append(cur_step_mean_IOU)
            # # 生成cam图用于保存
            # # 思路：20张cam图先取max，再归一化生成  |  先归一化再取max
            # cam_show, cam_indices = cam_show.max(dim=1)
            # cam_show = cam_show.squeeze(0)
            # cam_normed = cam_norm(cam_show)     # 归一化
            # oriImgpath = oriImgFolderpath+str(name[0])+'.jpg'
            # img = cv2.imread(oriImgpath)
            # height, width, _ = img.shape
            # # 生成可视化热力图
            # heatmap = cv2.applyColorMap(cv2.resize(cam_normed, (width, height)), cv2.COLORMAP_JET)
            # result = heatmap * 0.3 + img * 0.5
            # save_cam_path = validate_cam_path+str(name[0])+'_cam.jpg'
            # cv2.imwrite(save_cam_path, result)
            # cam-------------------------------------------------------------------------------------------------------

            # 根据mask生成pred_seg
            high_threshold = 0.4
            # print(type(mask))

            # print(name)
            # print(mask)
            mask[mask < high_threshold] = 0
            mask[mask >= high_threshold] = max_pred_cls+1
            # mask = mask.to(device)
            att_segPred = torch.as_tensor(mask).to(device)
            # print(att_segPred.dtype)
            att_segPred = att_segPred.int()
            # 计算mIOU
            confmat.update(seg_labels.flatten(), att_segPred.flatten())
            cur_step_mean_IOU = confmat.get_mIOU()
            validate_IOU.append(cur_step_mean_IOU)

            # 生成seg_label图片用于保存
            pseudo_res = torch.as_tensor(att_segPred, dtype=torch.uint8)
            toimg = transforms.ToPILImage()
            mask = toimg(pseudo_res)
            mask.putpalette(pallette)
            save_name = seg_res_path+str(name[0])+"_validate_seg_result.png"
            mask.save(save_name)

            data_loader.desc = "[test step {}] cur_step_mAP: {:.3f} all_step_mAP: {:.3f} cur_step_mean_iou:{:.3f} "\
                .format(step,
                        mean_ap,
                        mean_ap_all,
                        cur_step_mean_IOU)
        iou_sum = 0.0
        for item in validate_IOU:
            iou_sum += item
        validate_mean_IOU = iou_sum / len(validate_IOU)
        print(f'validate_mean_IOU: {validate_mean_IOU}')

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
