import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate
from voc12.data import VOC12ClsDataset
import datetime
import random
import numpy as np
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建模型权重保存文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 创建原始CAM保存文件夹
    if os.path.exists("./origincams") is False:
        os.makedirs("./origincams")

    # 用来保存训练以及验证过程中信息
    training_log_txt = "training_log_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

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
                                    transform=data_transform["train"],
                                    seg_label_flag=False)

    val_dataset = VOC12ClsDataset(img_name_list_path=args.train_img_name_path,
                                  voc12_root=args.dataset_path,
                                  transform=data_transform["val"],
                                  seg_label_flag=True)

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
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # print(model)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits or ~('pre_logits.fc.weight' in weights_dict)\
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))


    if args.freeze_layers:
        print('freeze')
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = create_optimizer(args, model)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    linear_scaled_lr = args.lr * args.batch_size / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    # loss_scaler = NativeScaler()
    lrate_scheduler, _ = create_scheduler(args, optimizer)

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
        mAP_multiple_class_label = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch, num_classes=args.num_classes)

        # scheduler.step()
        lrate_scheduler.step(epoch)

        tags = ["train_loss", "f1_score", "mAP_multiple_class_label", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], f1_score, epoch)
        tb_writer.add_scalar(tags[2], mAP_multiple_class_label, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # write into txt
        with open(training_log_txt, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_and_validate_log = f"[epoch: {epoch}]\n" \
                                     f"train_loss: {train_loss:.3f}     " \
                                     f"f1_score: {f1_score:.5f}     " \
                                     f"mAP_multiple_class_label: {mAP_multiple_class_label:.5f}     " \
                                     f"lr: {optimizer.param_groups[0]['lr']:.6f}\n"
            f.write(train_and_validate_log + "\n\n")

        # 保存最佳loss时权重
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), "./weights/{}-cur_ep{}-bestloss.pth".format(str(date), epoch))
            min_train_loss = train_loss

    # 保存最终epoch时权重
    torch.save(model.state_dict(), "./weights/{}-cur_ep{}-final.pth".format(str(date), epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('--model_name', type=str, default='vit_base', required=True, help='create model name')
    parser.add_argument('--num_classes', type=int, default=20, required=True)
    parser.add_argument('--weights', type=str, default='', required=False,
                        help='initial weights path, set to null character if you do not want to load weights')
    parser.add_argument('--freeze_layers', type=bool, default=False, required=False, help='F:freeze weight')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=1000, required=True)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # 文件路径
    parser.add_argument('--dataset_path', type=str,
                        default="/data/c425/tjf/datasets/VOC2012/", required=True)
    parser.add_argument('--train_img_name_path', type=str,
                        default="/data/c425/tjf/vit/voc12/train.txt", required=True)
    parser.add_argument('--val_img_name_path', type=str,
                        default="/data/c425/tjf/vit/voc12/val.txt", required=True)
    parser.add_argument('--ori_cam_path', type=str,
                        default="/data/c425/tjf/vit/origincams/", required=True)
    # 设备参数
    parser.add_argument('--device', default='cuda:1', type=str, required=True, help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    same_seeds(0)   # 随机化种子

    main(opt)
