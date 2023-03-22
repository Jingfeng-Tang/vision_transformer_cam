import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from torchvision import transforms
from torchvision.transforms import functional as F

IMG_FOLDER_NAME = "JPEGImages"
SEG_LABEL_FOLDER_NAME = "SegmentationClass"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def get_seg_label_path(img_name, voc12_root):
    return os.path.join(voc12_root, SEG_LABEL_FOLDER_NAME, img_name + '.png')


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    #img_name_list = img_gt_name_list
    return img_name_list


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None, seg_label_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.seg_label_flag = seg_label_flag

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.seg_label_flag:
            seg_label = PIL.Image.open(get_seg_label_path(name, self.voc12_root))
            seg_label = torch.as_tensor(np.array(seg_label), dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        if self.seg_label_flag:
            return name, img, seg_label
        else:
            return name, img


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None, seg_label_flag=False):
        super().__init__(img_name_list_path, voc12_root, transform, seg_label_flag)
        self.seg_label_flag = seg_label_flag
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):

        if self.seg_label_flag:
            name, img, seg_label = super().__getitem__(idx)

        else:
            name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        if self.seg_label_flag:
            return name, img, label, seg_label
        else:
            return name, img, label


