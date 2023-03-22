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
    mAP = []
    confmat = ConfusionMatrix(20)
    data_loader = tqdm(val_loader, file=sys.stdout)
    tags = ["mAP_multiple_class_label", "mIOU"]
    with torch.no_grad():
        validate_IOU = []
        for step, data in enumerate(data_loader):
            name, image, target, seg_labels = data
            # print(name)
            # seg_labels为原图大小，dtype为int64
            b, h, w = seg_labels.shape
            image, target, seg_labels = image.to(device), target.to(device), seg_labels.to(device)
            output, cams = model(image)
            cams_hw = cams.reshape(b, 14, 14, 20).permute(0, 3, 1, 2)
            cam_show = cams_hw
            #
            # print(cams_hw[0][0])
            # 先sigmoid试一试
            cams_hw = torch.sigmoid(cams_hw)
            # 归一化
            # for i in range(cams_hw.shape[0]):
            #     for j in range(cams_hw.shape[1]):
            #         cams_hw[i][j] = cams_hw[i][j] - torch.min(cams_hw[i][j])
            #         cams_hw[i][j] = cams_hw[i][j] / torch.max(cams_hw[i][j])

            output = torch.sigmoid(output)      # class tokens用于多标签分类
            # 计算mAP
            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            mean_ap = np.mean(mAP_list)
            mean_ap_all = np.mean(mAP)
            # 计算mIOU
            # 将20个类别的14*14cam插值回原图大小
            cam_label = torch.nn.functional.interpolate(cams_hw, size=(h, w), mode='bilinear', align_corners=False)
            # 设置阈值，将每个类的最大激活区域找出来，其他区域置0
            high_threshold = 0.916
            # print(cam_label[0][0])
            cam_label[cam_label < high_threshold] = 0
            # 添加背景类
            bg_label = torch.full((b, 1, h, w), 1e-5, device=device)
            # 整合背景类与目标类
            cam_bg_obj = torch.cat((bg_label, cam_label), dim=1)
            # cam生成的分割预测
            cam_segPred = torch.argmax(cam_bg_obj, dim=1)
            # 计算mIOU
            confmat.update(seg_labels.flatten(), cam_segPred.flatten())
            cur_step_mean_IOU = confmat.get_mIOU()
            validate_IOU.append(cur_step_mean_IOU)
            # 生成cam图用于保存
            # 思路：20张cam图先取max，再归一化生成  |  先归一化再取max
            cam_show, cam_indices = cam_show.max(dim=1)
            cam_show = cam_show.squeeze(0)
            cam_normed = cam_norm(cam_show)     # 归一化
            oriImgpath = oriImgFolderpath+str(name[0])+'.jpg'
            img = cv2.imread(oriImgpath)
            height, width, _ = img.shape
            # 生成可视化热力图
            heatmap = cv2.applyColorMap(cv2.resize(cam_normed, (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            save_cam_path = validate_cam_path+str(name[0])+'_cam.jpg'
            cv2.imwrite(save_cam_path, result)



            # a = []
            # b = a[10]

            # 生成seg_label图片用于保存
            pseudo_res = torch.as_tensor(cam_segPred, dtype=torch.uint8)
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
    parser.add_argument('--weights', type=str, default='/data/c425/tjf/vit/weights_pretarined_ep20/2023-03-19-cur_ep199-bestloss.pth', required=False,
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

    # a = torch.tensor(([1, 2, 5], [6, 7, 8]))
    # print(a.shape)
    # b = a.max(dim=1)
    # print(b)
    # a = []
    # b = a[10]

    val(opt)
