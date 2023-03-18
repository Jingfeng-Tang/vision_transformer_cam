import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate
from voc12.data import VOC12ClsDataset
import datetime
import random


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建模型权重保存文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 创建原始CAM保存文件夹
    if os.path.exists("./origincams") is False:
        os.makedirs("./origincams")

    tb_writer = SummaryWriter()

    # 遵循SEAM数据增强操作  目前这样数据增强只是为了可视化，使得cam图map到原图的相应位置
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # dataset
    train_dataset = VOC12ClsDataset(img_name_list_path=args.train_img_name_path,
                                    voc12_root=args.dataset_path,
                                    transform=data_transform["train"])

    val_dataset = VOC12ClsDataset(img_name_list_path=args.train_img_name_path,
                                  voc12_root=args.dataset_path,
                                  transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    model = create_model(num_classes=20, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    min_train_loss = 100000.0
    date = datetime.date.today()
    print((str(date)))
    for epoch in range(args.epochs):
        # train
        train_loss, f1_score = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch)
        # validate
        evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch, num_classes=20)
        # print(str(confmat))

        scheduler.step()

        tags = ["train_loss", "f1_score", "learning_rate", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], f1_score, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), "./weights/{}-cur_ep{}-bestloss.pth".format(str(date), epoch))
            min_train_loss = train_loss
    torch.save(model.state_dict(), "./weights/{}-cur_ep{}-final.pth".format(str(date), epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--dataset_path', type=str,
                        default="/data/c425/tjf/datasets/VOC2012/")
    parser.add_argument('--train_img_name_path', type=str,
                        default="/data/c425/tjf/vit/voc12/train.txt")
    parser.add_argument('--val_img_name_path', type=str,
                        default="/data/c425/tjf/vit/voc12/val.txt")
    parser.add_argument('--ori_cam_path', type=str,
                        default="/data/c425/tjf/vit/origincams/")
    parser.add_argument('--model-name', default='vit_base', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # parser.add_argument('--weights', type=str, default='./jx_vit_base_patch16_224_in21k-e5005f0a.pth',
    #                     help='initial weights path')
    # parser.add_argument('--weights', type=str, default='./model-499.pth',
    #                     help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    random.seed(0)  # 保证随机结果可复现
    main(opt)
