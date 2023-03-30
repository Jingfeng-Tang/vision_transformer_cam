"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
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
        attn = (q @ k.transpose(-2, -1)) * self.scale       # scale 0.125  attn: 8*12*197*197
        # attn 每个patch和patch之间的注意力
        attn = attn.softmax(dim=-1)     # attn： 8*12*197*197
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


class mask_Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(mask_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, mask_indices):
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
        attn = (q @ k.transpose(-2, -1)) * self.scale       # scale 0.125  attn: 8*12*197*197
        # attn 每个patch和patch之间的注意力
        attn = attn.softmax(dim=-1)     # attn： 8*12*197*197
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

    def forward(self, x):
        # step1: norm1 维度不变 8*197*768
        # step2: attn 注意力机制 还是8*197*768
        # step3: + 残差 8*197*768
        o, weights = self.attn(self.norm1(x))
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
        self.patch_d1 = torch.nn.Conv2d(768, 384, kernel_size=1)
        self.patch_d2 = torch.nn.Conv2d(384, 192, kernel_size=1)
        self.patch_d3 = torch.nn.Conv2d(192, 96, kernel_size=1)
        self.patch_d4 = torch.nn.Conv2d(96, 48, kernel_size=1)
        self.patch_d5 = torch.nn.Conv2d(48, 24, kernel_size=1)
        self.patch_d6 = torch.nn.Conv2d(24, 12, kernel_size=1)
        self.patch_d7 = torch.nn.Conv2d(12, 6, kernel_size=1)
        self.patch_d8 = torch.nn.Conv2d(6, 1, kernel_size=1)
        self.dim_reduct = nn.Linear(196, 20)
        self.is_train = is_train

    def forward_features(self, x):
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
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            if i == 5:
                # 如果是第五个block，那么mask背景
                j = 6
            if len(self.blocks) - i <= 12:
                attn_weights.append(weights_i)
                attn_matrix.append(x)

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
            return self.pre_logits(x[:, 0]), cams, attn_weights, attn_matrix     # cls token, cams, attn_weights:list 12stage attention blocks weights
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
        x, cams, attn_weights, attn_matrix = self.forward_features(x)

        objpatcht, dim_reduct_weight = self.forward_block5(attn_weights, attn_matrix, self.is_train)   # batchsize*20
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)

        return x, cams, attn_weights, attn_matrix, objpatcht, dim_reduct_weight


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
