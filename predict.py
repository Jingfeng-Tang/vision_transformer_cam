import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
import numpy as np
import cv2
from voc12.data import load_image_label_list_from_npy, load_img_name_list
import torch.nn.functional as F
torch.set_printoptions(threshold=np.inf)


name_list = load_img_name_list("/data/c425/tjf/vit/voc12/train.txt")
labels = load_image_label_list_from_npy(name_list)


def bitget(byteval, idx):
    return (byteval & 1 << idx) != 0  # 判断输入字节的idx比特位上是否为1


def color_map(N=256, normalized=False):
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        c = i
        r = g = b = 0  # 将类别索引和rgb分量都视为8位2进制数，即一个字节
        for j in range(8):  # 从高到低填入rgb分量的每个比特位
            r = r | bitget(c, 0) << (7 - j)  # 每次将类别索引的第0位放置到r分量
            g = g | bitget(c, 1) << (7 - j)  # 每次将类别索引的第1位放置到g分量
            b = b | bitget(c, 2) << (7 - j)  # 每次将类别索引的第2位放置到b分量
            c = c >> 3  # 将类别索引移位
        cmap[i] = np.array([r, g, b])
    cmap = cmap / 255 if normalized else cmap
    return cmap


# predict  单张图片预测
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transform_showimg = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.CenterCrop(224)
         ]
       )

    # load image
    img_path = "/data/c425/tjf/datasets/VOC2012/JPEGImages/2007_000491.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    imgo = Image.open(img_path)
    plt.imshow(imgo)
    # [N, C, H, W]
    img = data_transform(imgo)
    img_show = data_transform_showimg(imgo)
    # unloader = transforms.ToPILImage()
    # img_s = unloader(img_show)
    plt.imshow(img_show)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # print(f'cls_indict:{class_indict}')

    # 载入调色板
    palette_path = "./palette.json"
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # print(pallette)
    voc12pallettemap = color_map(21)
    # print(voc12pallettemap)

    # create model
    model = create_model(num_classes=20, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/2023-03-17-cur_ep485-bestloss.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # load label

    model.eval()
    # 获取模型的参数
    # print(model)
    # for param in model.parameters():
    #     print(f'param: {param.size()}')
    #     print('\n')
    # print('-------------------')
    # print(f'param[-1]: {model.parameters()}')
    with torch.no_grad():
        # predict class
        output, cams = model(img.to(device))
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        cams = cams.squeeze(0).reshape(14, 14, 20).permute(2, 0, 1)
        cam = cams[predict_cla]
        cam_t = cam
        cam = cam.cpu()
        cam_np = np.array(cam)
        # np 归一化
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        cam_np = np.uint8(255 * cam_np)
        # tensor 归一化
        cam_t = cam_t - torch.min(cam_t)
        cam_t = cam_t / torch.max(cam_t)
        cam_t = cam_t.unsqueeze(0)
        cam_t = cam_t.unsqueeze(0)
        # print(cam_t)
        # cam_t = torch.uint8(255 * cam_t)
        # print(cam_t)
        # 获得原图与尺寸
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        # 插值生成 pseudo res
        pseudo_res = F.interpolate(cam_t, size=(height, width), mode='bilinear', align_corners=False)
        # print(pseudo_res)
        # pseudo_res ×255   tensro 转int
        threshold = 0.6
        pseudo_res[pseudo_res < 0.6] = 0  # 生成掩模，背景部分为0
        pseudo_res[pseudo_res >= 0.6] = 10  # 生成掩模，背景部分为0
        # 此时目标为数值，背景为0
        # plattle
        pseudo_res = pseudo_res.squeeze(0)
        pseudo_res = pseudo_res.squeeze(0)
        pseudo_res = torch.tensor(pseudo_res, dtype=torch.uint8)
        print(pseudo_res.dtype)
        # pseudo_res_np = pseudo_res.cpu().detach().numpy()
        toimg = transforms.ToPILImage()
        mask = toimg(pseudo_res)
        # mask = Image.fromarray(pseudo_res_np)
        # print(mask)
        mask.putpalette(pallette)
        mask.save("test_result.png")

        # 生成可视化热力图
        heatmap = cv2.applyColorMap(cv2.resize(cam_np, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('CAM.jpg', result)

    str_label = 'GT labels: '
    for i in range(labels[8].shape[0]):
        if labels[8][i] == 1:
            str_label = str_label + class_indict[str(i)] + ' '
    plt.text(10, 10, str_label, fontsize=10, color='green')

    str_pred = ''
    labels_sum = labels[8].sum()
    labels_sum = labels_sum.item()
    labels_sum = int(labels_sum)

    val, index = torch.topk(predict, labels_sum, dim=0)
    for i in range(labels_sum):
        str_pred = "{}:{:.3}".format(class_indict[str(index[i].item())],
                                                  val[i]) + str_pred + ' '
    plt.title(str_pred)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
