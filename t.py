# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json


import matplotlib.pyplot as plt
import scipy.fft as fp
from scipy import fftpack

# input image
LABELS_file = 'imagenet-simple-labels.json'
img_name = '2009_001360'
image_file = '/data/c425/tjf/datasets/VOC2012/JPEGImages/'+img_name+'.jpg'


# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()
print(net)

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# print(net)
# print(net._modules.get('backbone'))


net._modules.get(finalconv_name).register_forward_hook(hook_feature)


# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    print(f'bz:{bz}, nc={nc}, h={h}, w={w}')
    output_cam = []
    for idx in class_idx:
        # idx对应的那个概率最大的类，所以是1*512
        # softmax 1*512               512*49       1*49
        print(idx)
        idx = 366
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  #feature conv 1*512*7*7  nc 512   h*w 49
        # print(f'camshape{cam.shape}')
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)     # 减去最小，除以最大，目的是归一化
        cam_img = cam / np.max(cam)
        print(f'cam_img_dtype{cam_img.dtype}')
        cam_img = np.uint8(255 * cam_img)
        # print(f'cam_imgshape{cam_img.shape}')
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# load the imagenet category list
with open(LABELS_file) as f:
    classes = json.load(f)


h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction

# 最后一层（layer4）输出的特征图 1*512*7*7
# print(f'type(features_blobs[0])={features_blobs[0].shape}')
# print(f'features_blobs[0]={features_blobs[0]}')

# 最后一层（layer4）输出的weight_softmax 1000*512       1000个类 对应的512通道
# print(f'type(weight_softmax)={weight_softmax.shape}')
# print(f'weight_softmax[0]={weight_softmax}')


for i in range(5):
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[i]])
    # render the CAM and output
    print('output CAM.jpg for the prediction: %s'%classes[idx[i]])
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(str(i)+'_CAM.jpg', result)



