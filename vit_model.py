"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # print(f'1x.shape:{x.shape}')
        # 8*3*224*224
        x = self.proj(x)
        # 8*768*14*14
        # print(f'2x.shape:{x.shape}')
        x = x.flatten(2).transpose(1, 2)
        # 8*196*768
        # print(f'3x.shape:{x.shape}')
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, current_layer, mask_indices):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape       # 8*197*768

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv 3* batchsize* 12* 197* 64
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print(f'q.shape:{q.shape}')
        # q 8* 12* 197* 64
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        if current_layer <= 4:
            attn = (q @ k.transpose(-2, -1)) * self.scale       # scale 0.125  attn: bs*12*197*197
        else:
            # mask_indices bs*197
            attn = (q @ k.transpose(-2, -1)) * self.scale  # scale 0.125  attn: bs*12*197*197
            # print('qkv')
            # print(f'attn.shape: {attn.shape}')
            # print(mask_indices.shape)
            mask_indices = mask_indices.unsqueeze(1).repeat(1, 12, 1, 1)
            # print(mask_indices.shape)
            # print(mask_indices[0, 0, 1, 0:])
            # print(attn[0, 0, 1, 0:])
            attn = attn + mask_indices
            # print(attn[0, 0, 1, 0:])
            # print(attn[0][0])
        # attn 每个patch和patch之间的注意力
        attn = attn.softmax(dim=-1)     # attn： 8*12*197*197
        # if current_layer == 5:
        #     for i in range(12):
        #         attn_show = attn[0][i].cpu().numpy()
        #         plt.subplot(3, 4, i+1)
        #         plt.imshow(attn_show)
        #     plt.show()
        #
        #     for i in range(12):
        #         # 每个头的cls与patch的注意力权重显示
        #         attn_show_cp = attn[0][i][0].unsqueeze(0).repeat(20, 1).cpu().numpy()
        #         plt.subplot(3, 4, i + 1)
        #         plt.imshow(attn_show_cp)
        #     plt.show()

        weights = attn
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # # attn： 8*12*197*197      v 8* 12* 197* 64
        x = (attn @ v)      # 8*12*197*64
        # print(f'x1.shape:{x.shape}')
        x = x.transpose(1, 2)      # 8*197*12*64        197个patch（cls）
        # print(f'x2.shape:{x.shape}')
        x = x.reshape(B, N, C)      # 8*197*768
        # print(f'x3.shape:{x.shape}')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(f'x1.shape:{x.shape}')
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # print(f'x2.shape:{x.shape}')
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, current_layer, mask_indices):
        # step1: norm1 维度不变 8*197*768
        # step2: attn 注意力机制 还是8*197*768
        # step3: + 残差 8*197*768
        o, weights = self.attn(self.norm1(x), current_layer, mask_indices)
        x = x + self.drop_path(o)
        # step1: norm2 维度不变 8*197*768
        # step2: mlp 注意力机制 还是8*197*768
        # step3: + 残差 8*197*768
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # end： 8*197*768
        return x, weights


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, is_train=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.pooling = F.adaptive_max_pool2d    # tang

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        self.twelveblocks = []
        self.patch_d1 = torch.nn.Conv2d(768, 256, kernel_size=1)
        self.patch_d2 = torch.nn.Conv2d(256, 32, kernel_size=1)
        self.patch_d3 = torch.nn.Conv2d(32, 1, kernel_size=1)

        self.norm1 = norm_layer(256)
        self.norm2 = norm_layer(32)

        self.hwp_map_labels = nn.Linear(16, 20)

        self.patch_d4 = torch.nn.Conv2d(96, 48, kernel_size=1)
        self.patch_d5 = torch.nn.Conv2d(48, 24, kernel_size=1)
        self.patch_d6 = torch.nn.Conv2d(24, 12, kernel_size=1)
        self.patch_d7 = torch.nn.Conv2d(12, 6, kernel_size=1)
        self.patch_d8 = torch.nn.Conv2d(6, 1, kernel_size=1)
        self.dim_reduct = nn.Linear(196, 20)

        self.head1 = nn.Linear(self.num_features, num_classes)

        self.relu = nn.ReLU()
        # self.patch_d1 = torch.nn.Conv2d(768, 256, kernel_size=1)

        self.is_train = is_train

    def forward_features(self, x):
        batchsize = x.shape[0]
        # print(f'label:{label.shape}')
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)       # 扩张cls token数量到batch size
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]       # 将cls token和patch token按维度1，拼接在一起
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)       # 加入位置编码

        # x = self.blocks(x)                  # 进入transformer block  此时x 8*197*768
        attn_weights = []
        attn_matrix = []
        mask_indices = torch.zeros((batchsize, 197, 197))
        for i, blk in enumerate(self.blocks):       # 原来12个block，现在5个
            x, weights_i = blk(x, i, mask_indices)
            if len(self.blocks) - i <= 12:
                attn_weights.append(weights_i)
                attn_matrix.append(x)
            if i >= 4:
                # print(weights_i.shape)        #  16*12*197*197
                # 根据第5个weights_i获取mask
                # weights_i是第五个block的权重
                att_mat = torch.mean(weights_i, dim=1)  # 16 * 197 * 197: block * batchsize * patches * patches
                # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
                residual_att = torch.eye(att_mat.size(2)).cuda()  # 197 * 197 identity matrix
                aug_att_mat = att_mat + residual_att
                aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # 16 * 197 * 197
                mask_i = aug_att_mat[:, 0, 1:]  # 16*196
                mask_14 = mask_i / mask_i.max()  # 16*196
                # 获取小于0.25的权重的索引
                patchTokensIndex = []
                for j in range(int(mask_14.shape[0])):
                    patchid = torch.lt(mask_14[j], 0.25)
                    patchid = patchid.int()
                    patchTokensIndex.append(patchid)
                mask_indices = torch.stack(patchTokensIndex).squeeze(1)  # 16*196

                cls_zero = torch.zeros((batchsize, 1)).cuda()
                mask_indices = torch.cat((cls_zero, mask_indices), dim=1)       # 16 * 197
                mask_indices = mask_indices.unsqueeze(2).repeat(1, 1, 197)      # 16 * 197 * 197
                mask_indices = mask_indices + mask_indices.permute(0, 2, 1)
                # print(f'mask_indices: {mask_indices.shape}')
                mask_indices[mask_indices > 1] = 1  # 将相加后的2变成1
                # print(mask_indices.shape)

                # print(mask_indices)
                # print(str(i+1)*50)
                # print(f'当前block被遮住的背景patch个数: {mask_indices.sum(1).item()}')
                # perdict 可视化使用-------------------------------------------------------------------------------------
                # 0-255颜色指示
                # indictor_0_255 = torch.arange(0, 255).unsqueeze(0).repeat(20, 1).cpu().numpy()
                # plt.subplot(3, 2, 3)
                # plt.imshow(indictor_0_255)
                # plt.title('indictor_0_255')
                # mask_indices_np = mask_indices.cpu().numpy()
                # plt.subplot(3, 2, 1)
                # plt.imshow(mask_indices_np)

                # mask_indices = mask_indices.unsqueeze(2).repeat(1, 1, 196)  # 1*196*196
                # mask_indices_np1 = mask_indices.permute(1, 2, 0).cpu().numpy()
                # plt.subplot(3, 2, 2)
                # plt.imshow(mask_indices_np1)
                # plt.title(str(i+1))

                # perdict 可视化使用-------------------------------------------------------------------------------------
                # mask_indices = mask_indices + mask_indices.permute(0, 2, 1)
                # mask_indices_np_per_add = mask_indices.permute(1, 2, 0).cpu().numpy()
                # plt.subplot(3, 2, 4)
                # plt.imshow(mask_indices_np_per_add)

                # 这种只屏蔽了patch与patch之间的交互，没有阻止patch与cls token之间的信息交互
                # cls_1 = torch.zeros((batchsize, 1, 196)).cuda()
                # mask_indices = torch.cat((cls_1, mask_indices), dim=1)
                # cls_1_1 = torch.zeros((batchsize, 197, 1)).cuda()
                # mask_indices = torch.cat((cls_1_1, mask_indices), dim=2)
                # plt.subplot(3, 2, 5)
                # plt.imshow(mask_indices.permute(1, 2, 0).cpu().numpy())

                mask_indices = -100 * mask_indices
                # print(mask_indices.shape)
                # print(mask_indices[0][0])
                # plt.subplot(3, 2, 6)
                # plt.imshow(mask_indices.permute(1, 2, 0).cpu().numpy())
                # plt.show()

        # 最后一个block，把注意力权重较高的patch挑出来，我认为这些patch是一种聚类原型，将它们与标签对齐，进行训练。
        # 这样在eval阶段，对高权重目标patch进行预测可以获得其对应的类别，这些高权重目标patch与其相同类的patch之间的语义相似度
        # 也非常高
        att_mat = torch.mean(weights_i, dim=1)  # 16 * 197 * 197: block * batchsize * patches * patches
        # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(2)).cuda()  # 197 * 197 identity matrix
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # 16 * 197 * 197
        mask_i = aug_att_mat[:, 0, 1:]      # 16*196
        mask_14 = mask_i / mask_i.max()     # 16*196
        # print(f'x.shape: {x.shape}')      # 16 * 197 * 768
        allbs_hw_patch = []
        for j in range(batchsize):
            # 取top16个patch的index
            val, index = torch.topk(mask_14[j], 16, dim=0)
            # high_weight_patch16 = torch.zeros((1, 768), device='cuda:1')     # bs*1*768
            index = index.squeeze(0)
            # print(f'index.shape: {index.shape}')
            high_weight_patch16 = x[j][index[0]+1].unsqueeze(0)
            # print(f'high_weight_patch16.shape: {high_weight_patch16.shape}')
            for i in range(15):
                # print(i)
                # print(f'high_weight_patch16.shape: {high_weight_patch16.unsqueeze(0).shape}')
                # print(f'x[j][index[i+1]+1: {x[j][index[i+1]+1].unsqueeze(0).shape}')
                # print('-------')
                high_weight_patch16 = torch.cat((high_weight_patch16, x[j][index[i+1]+1].unsqueeze(0)), dim=0)  # 17*768
                # print(f'cat after high_weight_patch16.shape: {high_weight_patch16.shape}')
            allbs_hw_patch.append(high_weight_patch16)
        allbs_hw_p_ts = torch.stack(allbs_hw_patch).squeeze(1)
        #  print(allbs_hv_p_ts.shape)   # 16*16*768  bs* hwp*features
        ori_allbs_hw_p_ts = allbs_hw_p_ts       # 16*16*768  bs* hwp * features
        allbs_hw_p_ts = allbs_hw_p_ts.mean(dim=1)       # 16*768
        allbs_hw_p_ts = self.head1(allbs_hw_p_ts)       # 768 20

        predcls = torch.sigmoid(allbs_hw_p_ts)
        print(f'predcls: {predcls}')
        predcls[predcls >= 0.9] = 1
        predcls[predcls < 0.9] = 0
        # print(f'predcls: {predcls}')
        # print(f'predcls: {predcls.shape}')      # 16*20
        if not self.is_train:
            zero_t = torch.zeros((1, 768), device='cuda:1')
            for l in range(batchsize):
                print(f'predcls: {predcls}')
                clsh1_weight = self.head1.weight.data  # 20*768
                for k in range(20):
                    if predcls[l][k] == 0:      # 如果不是模型所预测的那个类
                        clsh1_weight[k] = zero_t        # 将所属权重置0
                # clsh1_softmax = torch.softmax(clsh1_weight, dim=1)
                # cls_to_768 = torch.argmax(clsh1_softmax, dim=0)      # 为768个特征赋予类别
                # print(f'cls_to_768.shape: {cls_to_768}')
                # 将768个特征与16个patch建立联系（为16个patch赋予类别）
                # curimg_ori_allbs_hw_p_ts = ori_allbs_hw_p_ts[l]
                # zero_16_768 = torch.arange(21, 12309, 1, device='cuda:1').reshape(768, 16)
                # # print(f'zero_16_768: {zero_16_768}')
                # # 每个特征哪个patch贡献大
                # contriPatchindex = torch.argmax(curimg_ori_allbs_hw_p_ts, dim=0)
                # # print(f'contriPatchindex: {contriPatchindex.shape}')  # 16*20
                # # print(f'contriPatchindex: {contriPatchindex}')
                # for m in range(768):
                #     zero_16_768[m][contriPatchindex[m]] = cls_to_768[m]
                # # print(f'zero_16_768.shape: {zero_16_768}')
                # patch_to_cls, indice = torch.mode(zero_16_768, dim=0)
                # # print(f'patch_to_cls.shape: {patch_to_cls.shape}')
                # # print(f'patch_to_cls: {patch_to_cls}')




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

        # 8*197*768
        x = self.norm(x)
        # -------------------------------------------------------------------------------------------------
        # 获得cam 测试
        patch_tokens = x[:, 1:]         # get patch_tokens
        # 8*196*768
        # get classifier weight
        # cls_patch = self.pooling(patch_tokens, (1, 1))
        # print(f'cls_patch.shape: {cls_patch.shape}')            # 8*1*1
        # cls_res = self.head(cls_patch)
        cls_weight = self.head.weight
        # print(f'self.head.weight.shape: {self.head.weight.shape}')
        # patch_tokens_test = patch_tokens.permute(0, 2, 1).reshape(8, 768, 14, 14)   # 输入
        # print(f'patch_tokens_test.shape: {patch_tokens.shape}')
        cls_weight_test = cls_weight
        # cls_weight_test = cls_weight_test.unsqueeze(dim=1).unsqueeze(dim=2)
        # cls_weight_test = cls_weight_test.repeat(8, 1, 1, 1)
        cls_weight_test = cls_weight_test.permute(1, 0)
        # print(f'cls_weight_test.shape: {cls_weight_test.shape}')  # 5*768
        cams = torch.einsum('ijk,kl->ijl', patch_tokens, cls_weight_test)
        # cam = F.conv2d(patch_tokens_test, cls_weight_test).detach()  # 8*768*14*14  cls.weight:
        # -------------------------------------------------------------------------------------------------
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), cams, attn_weights, attn_matrix, allbs_hw_p_ts     # cls token, cams, attn_weights:list 12stage attention blocks weights
        else:
            return x[:, 0], x[:, 1]

    def forward_block5(self, x, attn_matrix, is__train):
        if is__train:
            # x是第五个block的权重
            threshold = 0.6
            # first, you should return all attention matrix in self-attention model (12 stages), and then stack them.
            att_mat = torch.stack(x).squeeze(1)  # 12 * 16 * 12 * 197 * 197: block * batchsize * heads * patches * patches
            att_mat = torch.mean(att_mat, dim=2)  # 12 * 16 * 197 * 197: block * batchsize * patches * patches
            # To account for residual connections, then add an identity matrix to the attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(2)).cuda()  # 197 * 197 identity matrix
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)   # 12 * 16 * 197 * 197
            v_i = aug_att_mat[4]    # v_i 第五个block的所有样本的注意力权重   16 * 197 * 197
            mask_i = v_i[:, 0, 1:]         # 16*196
            mask_14 = mask_i / mask_i.max()     # 16*196
            # 获取大于0.25的权重的索引
            patchTokensIndex = []
            for i in range(int(mask_14.shape[0])):
                patchid = torch.gt(mask_14[i], 0.25)
                patchTokensIndex.append(patchid)
            patchTokensIndexT = torch.stack(patchTokensIndex).squeeze(1)        # 16*196
            # attn_matrix 16 * 197 * 768 去除cls token
            cls_patchTokens = attn_matrix[4]
            patchtokens = cls_patchTokens[:, 1:, :]     # 16*196*768
            # 将patchTokensIndexT扩充至768维度
            patchTokensIndexT = patchTokensIndexT.unsqueeze(2)
            patchTokensIndexT = patchTokensIndexT.repeat(1, 1, 768)     # 16*196*768
            objpatcht = torch.einsum("kij, kij -> kij", patchtokens, patchTokensIndexT)     # 16*196*768
            # 删除全零tensor 并补齐
            for i in range(objpatcht.shape[0]):
                nonZeroRows = torch.abs(objpatcht[i]).sum(dim=1) > 0
                # 删除全零tensor
                new_objpatcht = objpatcht[i][nonZeroRows]
                # 补齐
                # 分母出现0
                if new_objpatcht.shape[0] == 0:
                    noobj = torch.zeros((196, 768))
                    objpatcht[i] = noobj
                    continue
                # print(f'new_objpatcht.shape[0]: {new_objpatcht.shape[0]}')
                i_times = 196 // new_objpatcht.shape[0]
                j = 196 % new_objpatcht.shape[0]
                full_objpatcht = new_objpatcht
                for ai in range(i_times-1):
                    full_objpatcht = torch.cat([full_objpatcht, new_objpatcht])
                for aj in range(j):
                    full_objpatcht = torch.cat([full_objpatcht, new_objpatcht[aj].unsqueeze(0)])
                objpatcht[i] = full_objpatcht
            # 此时objpatcht 为全目标特征
            # print(objpatcht.shape)        #  16*196*768
            objpatcht = objpatcht.reshape(objpatcht.shape[0], 14, 14, 768).permute(0, 3, 1, 2)
            # print('*****************111111*************')
        else:
            # 去除cls token
            cls_patchTokens = attn_matrix[4]        # 1*197*768
            patchTokens = cls_patchTokens[:, 1:, :].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)     # 1*768*14*14
            objpatcht = patchTokens


        # 1*1 卷积降维
        objpatcht = self.patch_d1(objpatcht)
        objpatcht = self.patch_d2(objpatcht)
        objpatcht = self.patch_d3(objpatcht)
        objpatcht = self.patch_d4(objpatcht)
        objpatcht = self.patch_d5(objpatcht)
        objpatcht = self.patch_d6(objpatcht)
        objpatcht = self.patch_d7(objpatcht)
        objpatcht = self.patch_d8(objpatcht)
        objpatcht = objpatcht.reshape(objpatcht.shape[0], 196)
        objpatcht = self.dim_reduct(objpatcht)      #  16  20

        # print('self.dim_reduct.parameters()----------------------------------------------------')
        # for name, param in self.dim_reduct.named_parameters():
        #     print(name)
        #     print(param.shape)
        #     print('--param.shape')
        # # 会循环两次  20 196  20  只要20 196
        dim_reduct_weight = list(self.dim_reduct.named_parameters())[0]
        # 最后是要输出目标patch，实现目标语义对齐  0.25
        return objpatcht, dim_reduct_weight

    def forward(self, x):
        x, cams, attn_weights, attn_matrix, allbs_hw_p_ts = self.forward_features(x)

        # objpatcht, dim_reduct_weight = self.forward_block5(attn_weights, attn_matrix, self.is_train)   # batchsize*20
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            # # 获取768-20的权重
            # cls_head_weights = list(self.head.named_parameters())[0][1].data        # 20*768

        # return x, cams, attn_weights, attn_matrix, objpatcht, dim_reduct_weight
        return x, cams, attn_weights, attn_matrix, allbs_hw_p_ts


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
