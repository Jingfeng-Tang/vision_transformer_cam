import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
import numpy as np
import cv2

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_transform_showimg = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.CenterCrop(224)
         ]
       )

    # load image
    # img_path = "/data/c425/tjf/datasets/flower_photos/tulips/16907559551_05ded87fb2_n.jpg"
    # img_path = "/data/c425/tjf/datasets/flower_photos/sunflowers/4933822272_79af205b94.jpg"
    img_path = "/data/c425/tjf/datasets/flower_photos/daisy/3445110406_0c1616d2e3_n.jpg"
    # img_path = "/data/c425/tjf/datasets/flower_photos/dandelion/10200780773_c6051a7d71_n.jpg"
    # img_path = "/data/c425/tjf/datasets/flower_photos/roses/14176042519_5792b37555.jpg"
    # img_path = "./rose.jpg"
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

    # create model
    model = create_model(num_classes=5, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
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
        cams = cams.squeeze(0).reshape(14, 14, 5).permute(2, 0, 1)
        cam = cams[predict_cla]
        cam = cam.cpu()
        cam = np.array(cam)
        # 归一化
        cam = cam - np.min(cam)  # 减去最小，除以最大，目的是归一化
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        heatmap1 = cv2.applyColorMap(cv2.resize(cam_img, (14, 14)), cv2.COLORMAP_JET)
        cv2.imwrite('CAM_14.jpg', heatmap1)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('CAM.jpg', result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
