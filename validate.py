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
from utils import compute_mAP
from torchvision.transforms import functional as F
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


def val(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建原始CAM保存文件夹
    if os.path.exists("./origincams") is False:
        os.makedirs("./origincams")

    # 用来保存验证过程中信息
    val_log_txt = "testing_log_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


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
    # confmat = ConfusionMatrix(num_classes)
    data_loader = tqdm(val_loader, file=sys.stdout)
    tags = ["mAP_multiple_class_label", "mIOU"]
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            name, image, target, seg_labels = data
            # seg_labels为原图大小，dtype为int64
            image, target, seg_labels = image.to(device), target.to(device), seg_labels.to(device)
            output, cams = model(image)
            print(cams.shape)
            print(cams.dtype)
            print(cams.dtype)
            output = torch.sigmoid(output)      # class tokens用于多标签分类
            # 计算mAP
            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            mean_ap = np.mean(mAP_list)
            mean_ap_all = np.mean(mAP)
            # 计算mIOU

            data_loader.desc = "[test step {}] cur_step_mAP: {:.3f} all_step_mAP: {:.3f}".format(step,
                                                                                                 mean_ap,
                                                                                                 mean_ap_all)

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

    val(opt)
