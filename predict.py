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
from voc12.data import load_image_label_from_xml
import random
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)

name_list = load_img_name_list("/data/c425/tjf/vit/voc12/train.txt")
labels = load_image_label_list_from_npy(name_list)
CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

from matplotlib.colors import LinearSegmentedColormap
clist = [(1,0,0),(0,1,0),(0,0,1),(1,0.5,0),(1,0,0.5),(0.5,1,0)]
newcmp = LinearSegmentedColormap.from_list('chaos', clist)


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

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





# predict  单张图片预测
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建predict_cam保存文件夹
    predict_cam_path = './predict_cam/'
    if os.path.exists(predict_cam_path) is False:
        os.makedirs(predict_cam_path)

    data_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transform_showimg = transforms.Compose(
        [transforms.Resize([224, 224]),
         ]
       )

    # load image
    img_name = '2009_004434'
    img_path = '/data/c425/tjf/datasets/VOC2012/JPEGImages/'+img_name+'.jpg'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    imgo = Image.open(img_path).convert("RGB")
    ori_h = imgo.height
    ori_w = imgo.width
    # plt.imshow(imgo)
    # [N, C, H, W]
    img = data_transform(imgo)
    img_show = data_transform_showimg(imgo)
    # unloader = transforms.ToPILImage()
    # img_s = unloader(img_show)
    # plt.imshow(img_show)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 载入调色板
    palette_path = "./palette.json"
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    voc12pallettemap = color_map(21)



    # create model
    model = create_model(num_classes=20, has_logits=False).to(device)
    # load model weights
    # model_weight_path = "/data/c425/tjf/vit/weights_pretrained_ep1000/2023-03-21-cur_ep997-bestloss.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    # model_weight_path = "/data/c425/tjf/vit/weights_pretrained_ep1000_train/2023-03-24-cur_ep999-final.pth"
    model_weight_path = "/data/c425/tjf/vit/weights_head1/2023-04-05-cur_ep78-bestloss.pth"
    # model_weight_path = "/data/c425/tjf/vit/weights_pretrained_ep1000_freeze/2023-03-21-cur_ep999-final.pth"
    # model_weight_path = "/data/c425/tjf/vit/weights_8conv/2023-03-30-cur_ep787-bestloss.pth"
    weights_dict = torch.load(model_weight_path, map_location=device)
    del_keys = ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))

    # load label
    model.eval()
    model.is_train = False

    with torch.no_grad():
        # predict class
        # output, cams, attn_w, attn_m, objpatcht, drweight = model(img.to(device))
        output, cams, attn_w, attn_m, allbs_hw_p_ts = model(img.to(device))
        # objpatcht = torch.sigmoid(objpatcht)
        # print(objpatcht)
        # # block5 attention map------------------------------------------------------------------------------------------
        # att_mat = torch.stack(attn_w).squeeze(1)  # 12 * 12 * 197 * 197: block * heads * patches * patches
        # att_mat = torch.mean(att_mat, dim=1)  # 12 * 197 * 768: block * patches * embeddings  在heads维度取平均
        # # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
        # residual_att = torch.eye(att_mat.size(1))  # 197 * 197 单位矩阵
        # att_mat = att_mat.cpu()     # 12*197*197
        # aug_att_mat = att_mat + residual_att
        # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)   # 12*197*197
        # attn_weigths = aug_att_mat[4]   # 197*197 第五个block
        # # 取196
        # mask_i = attn_weigths[0, 1:].detach().numpy()
        # mask_14 = mask_i / mask_i.max()     #  vit给出的196patch的归一化权重
        # # 可视化mask_14 attn_weight
        # threshold = 0.2
        # mask_14_seg = mask_14
        # mask_14_seg[mask_14_seg<threshold] = 0      # 小于阈值，归为背景     # 196
        #
        # # cnn_weight = drweight[1].data
        # # cnn_weight = torch.softmax(cnn_weight, dim=0)
        # # choosen_label = torch.argmax(cnn_weight, dim=0)
        # # for i in range(mask_14_seg.shape[0]):
        # #     if mask_14_seg[i] != 0:
        # #         mask_14_seg[i] = choosen_label[i]
        # mask_14_seg = torch.as_tensor(mask_14_seg, dtype=torch.uint8)
        # mask_14_seg = mask_14_seg.unsqueeze(0).reshape(1, 14, 14)
        # toimg = transforms.ToPILImage()
        # mask = toimg(mask_14_seg)
        # mask.putpalette(pallette)
        # mask.save("predict_mask_14_result.png")
        #
        # # mask_14 = torch.tensor(mask_14).unsqueeze(0).cuda()
        # # # 还差196对20的weights
        # # # drweight[1].data.shape            #  20*196   # cnn给出的每个patch对20个类的权重
        # # for i in range(20):
        # #     drweight[1].data[i] = drweight[1].data[i] / drweight[1].data[i].max()
        # #
        # # multipclsattmap = torch.einsum('ij, kj -> ij', drweight[1].data, mask_14)
        # # for i in range(20):
        # #     multipclsattmap[i] = multipclsattmap[i] / multipclsattmap[i].max()
        # # multipclsattmap = multipclsattmap.reshape(20, 14, 14)
        # # multipclsattmap = multipclsattmap.unsqueeze(0).permute(1, 0, 2, 3)
        # # multipclsattmap = F.interpolate(multipclsattmap, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        # # seg_rres = torch.argmax(multipclsattmap, dim=0)
        # # seg_rres = torch.as_tensor(seg_rres, dtype=torch.uint8)
        # # toimg = transforms.ToPILImage()
        # # mask = toimg(seg_rres)
        # # # mask = Image.fromarray(pseudo_res_np)
        # # # print(mask)
        # # mask.putpalette(pallette)
        # # mask.save("predict_test_result.png")
        # # block5 attention map------------------------------------------------------------------------------------------



        # attention map-------------------------------------------------------------------------------------------------
        # first, you should return all attention matrix in self-attention model (12 stages), and then stack them.
        att_mat = torch.stack(attn_w).squeeze(1)    # 12 * 12 * 197 * 197: block * heads * patches * patches
        att_mat = torch.mean(att_mat, dim=1)        # 12 * 197 * 197: block * patches * embeddings  在heads维度取平均
        for i in range(len(attn_m)):
            block_i_patch = attn_m[i]
            feature1 = block_i_patch
            feature2 = block_i_patch
            feature1 = F.normalize(feature1).squeeze(0)  # F.normalize只能处理两维的数据，L2归一化
            feature2 = F.normalize(feature2).squeeze(0)
            # print(feature2.shape)
            distance = feature1.mm(feature2.t())  # 计算余弦相似度
            distance_np = distance.cpu().numpy()
            # if i == 4:
            #     print(f'孩子头与猫头： {distance_np[52][88]}')
            #     print(f'孩子头与孩子腰部： {distance_np[52][121]}')
            #     print(f'猫腿与猫头： {distance_np[116][88]}')

            plt.subplot(7, 6, 3*i+1)        # 所有patch的余弦相似度可视化
            plt.imshow(distance_np)
            plt.xticks([])
            plt.yticks([])

        # 获得原图与尺寸
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))           # 768 * 768 初等矩阵
        att_mat = att_mat.cpu()
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        # print(aug_att_mat.shape)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space.
        v = joint_attentions[-1]

        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

        # cam_img = mask / np.max(mask)
        # cam_img = np.uint8(255 * cam_img)
        # # print(f'cam_imgshape{cam_img.shape}')
        # mask = cv2.resize(cam_img, img.size())
        # result = (mask * img).astype("uint8")
        # print(mask)
        # print(type(v.numpy()))

        # plt.subplot(4, 4, 13)
        # v_np = v.numpy()
        # v_show = v_np/v_np.max()
        # plt.imshow(v_show)

        mask = cv2.resize(mask / mask.max(), (width, height))[..., np.newaxis]
        result = (mask * img).astype("uint8")
        # plt.subplot(4, 4, 14)
        # plt.imshow(img)
        plt.subplot(7, 6, 42)
        plt.imshow(img)
        plt.imshow(mask*255, alpha=0.4, cmap='rainbow')
        plt.xticks([])
        plt.yticks([])

        # mask_14 = cv2.resize(mask/mask.max(), (14, 14))
        # plt.subplot(4, 4, 16)
        # plt.imshow(mask_14)

        result_12 = []
        for i in range(12):
            v_i = aug_att_mat[i]
            mask_i = v_i[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask_14 = mask_i / mask_i.max()
            mask_i = cv2.resize(mask_i / mask_i.max(), (width, height))[..., np.newaxis]
            # print('--------------------------------------')
            # result_12.append((mask_i * img).astype("uint8"))
            result_12.append((mask_i*255).astype("uint8"))

            plt.subplot(7, 6, 3*i+2)            # 14*14特征图
            plt.imshow(mask_14)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(7, 6, 3 * (i + 1))          # 原图加掩膜
            plt.imshow(img)
            plt.imshow(result_12[i], alpha=0.4, cmap='rainbow')
            plt.xticks([])
            plt.yticks([])

        # 原图
        plt.subplot(7, 6, 39)
        plt.imshow(img)

        # attention map--------------------------------------------------------------------------------------------------


        # output = torch.squeeze(output).cpu()
        # predict = torch.sigmoid(output)
        allbs_hw_p_ts = torch.squeeze(allbs_hw_p_ts).cpu()
        predict = torch.sigmoid(allbs_hw_p_ts)
        predict_cla = torch.argmax(predict).numpy()

        # cams----------------------------------------------------------------------------
        cams = cams.squeeze(0).reshape(14, 14, 20).permute(2, 0, 1)
        # 生成所有种类的热力图

        # # 插值生成 pseudo res
        # for i in range(20):
        #     cam_orisize = cams[i].cpu()
        #     cam_np = np.array(cam_orisize)
        #     # np 归一化
        #     cam_np = cam_np - np.min(cam_np)
        #     cam_np = cam_np / np.max(cam_np)
        #     cam_np = np.uint8(255 * cam_np)
        #     # 生成可视化热力图
        #     heatmap = cv2.applyColorMap(cv2.resize(cam_np, (width, height)), cv2.COLORMAP_JET)
        #     result = heatmap * 0.3 + img * 0.5
        #     cv2.imwrite(predict_cam_path + img_name+'_CAM_'+CAT_LIST[i]+'__'+str(predict[i].numpy())+'.jpg', result)


        # # print(cams.shape)
        # cam = cams[predict_cla]
        # cam_t = cam
        # cam = cam.cpu()
        # cam_np = np.array(cam)
        # # np 归一化
        # cam_np = cam_np - np.min(cam_np)
        # cam_np = cam_np / np.max(cam_np)
        # cam_np = np.uint8(255 * cam_np)
        # # tensor 归一化
        # cam_t = cam_t - torch.min(cam_t)
        # cam_t = cam_t / torch.max(cam_t)
        # cam_t = cam_t.unsqueeze(0)
        # cam_t = cam_t.unsqueeze(0)
        # # print(cam_t)
        # # cam_t = torch.uint8(255 * cam_t)
        # # print(cam_t)
        #
        # # 插值生成 pseudo res
        # pseudo_res = F.interpolate(cam_t, size=(height, width), mode='bilinear', align_corners=False)
        # # print(pseudo_res)
        # # pseudo_res ×255   tensro 转int
        # threshold = 0.6
        # pseudo_res[pseudo_res < 0.6] = 0  # 生成掩模，背景部分为0
        # pseudo_res[pseudo_res >= 0.6] = 10  # 生成掩模，背景部分为0
        # # 此时目标为数值，背景为0
        # # plattle
        # pseudo_res = pseudo_res.squeeze(0)
        # pseudo_res = pseudo_res.squeeze(0)
        # pseudo_res = torch.as_tensor(pseudo_res, dtype=torch.uint8)
        #
        # # pseudo_res_np = pseudo_res.cpu().detach().numpy()
        # toimg = transforms.ToPILImage()
        # mask = toimg(pseudo_res)
        # # mask = Image.fromarray(pseudo_res_np)
        # # print(mask)
        # mask.putpalette(pallette)
        # mask.save("predict_test_result.png")
        #
        # # 生成可视化热力图
        # heatmap = cv2.applyColorMap(cv2.resize(cam_np, (width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # cv2.imwrite('predict_CAM.jpg', result)

        # cams----------------------------------------------------------------------------

    label_name = load_image_label_from_xml(img_name, '/data/c425/tjf/datasets/VOC2012/')

    str_label = 'GT labels: '
    label_num_count = 0
    for i in range(label_name.shape[0]):
        if label_name[i] == 1:
            str_label = str_label + CAT_LIST[i] + ' '
            label_num_count += 1

    str_pred = ''

    val, index = torch.topk(predict, label_num_count, dim=0)
    for i in range(label_num_count):
        str_pred = "{}:{:.3}".format(class_indict[str(index[i].item())],
                                     val[i]) + str_pred + ' '

    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.text(-1500, -2400, str_label, fontsize=10, color='green')
    plt.text(-1500, -2300, str_pred, fontsize=10, color='black')

    plt.plot()
    plt.show()
    # plt.title(str_pred)


if __name__ == '__main__':
    # a = torch.tensor(([-1, -2, -2], [10, 1, 30], [12, 89, 89], [40, 50, 60], [70, 80, 90]))
    # a = torch.tensor(([1.1, 2.3], [1.1, 4.6]))
    a = torch.tensor([0,1])
    print(a)
    #b = torch.softmax(a, dim=0)   # 扩充768，这里3
    # a = a
    a[0]=12
    print(a)









    a = []
    b = a[10]

    same_seeds(0)
    main()
