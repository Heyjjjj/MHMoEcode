import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from DilatedStripConv import DynamicConv, DiletedConv
from SparseMoE import SparseMoE
from DEConv import DEConv
from einops import rearrange, repeat


# torch.autograd.set_detect_anomaly(True)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def init_weights(self):
        init.normal_(self.fc1.weight, std=0.001)
        init.constant_(self.fc1.bias, 0)
        init.normal_(self.fc2.weight, std=0.001)
        init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.expand(x.permute(0, 2, 3, 1))  # B, H, W, C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x).permute(0, 3, 1, 2)  # B, C, H, W

        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim,  num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, H, W, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MAMoE(nn.Module):
    "Implementation of Morphology-Aware MoE block"
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=1, extend_scope=1, if_offset=True, device='cuda'):
        super().__init__()
        head_dim = dim // 3
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        

        self.expert1 = nn.Conv2d(head_dim, head_dim, 3, padding=1, bias=True)
        self.expert2 = nn.Conv2d(head_dim, head_dim, 3, padding=2, dilation=2, bias=True)
        self.expert3 = DynamicConv(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=0,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.expert4 = DiletedConv(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=0,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.expert5 = DynamicConv(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=1,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.expert6 = DiletedConv(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=1,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.SparseMoE = nn.ModuleList([SparseMoE(head_dim, num_experts=5, top_k=1) for _ in range(3)])
        self.linear = nn.ModuleList([nn.Linear(head_dim*2, head_dim) for _ in range(3)])
        self.expert_moe1 = nn.ModuleList([self.expert1,self.expert3,self.expert5])
        self.expert_moe2 = nn.ModuleList([self.expert2,self.expert4,self.expert6])

        self.qkv = nn.ModuleList([nn.Linear(head_dim, dim, bias=qkv_bias) for i in range(3)])
        self.attention = nn.ModuleList([GlobalAttention(head_dim, num_heads, qkv_bias, qk_scale, attn_drop) for i in range(3)])
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.proj_1(x)
        x = x.reshape(B, H, W, 3, C // 3).permute(3, 0, 4, 1, 2)
        x_moe = [None] * 3
        qkv = [None] * 3
        #3 b h w c/3

        for i in range(3):
            x_moe[i] = torch.concat([self.expert_moe1[i](x[i].clone()),self.expert_moe2[i](x[i].clone())],1)
            x_moe[i] = self.linear[i](x_moe[i].permute(0,2,3,1))
            x_moe[i]= self.SparseMoE[i](x_moe[i])
            qkv[i] = self.qkv[i](x_moe[i]).reshape(B,H,W,3,C//3).permute(3,0,1,2,4)
            x[i] = self.attention[i](qkv[i][0],qkv[i][1],qkv[i][2]).permute(0,3,1,2)

        x = x.permute(1,3,4,0,2).reshape(B,H,W,C)
        x = self.proj_3(x)
        return x

class MSVF(nn.Module):
    "Implementation of Multi-Scale View Fusion block"
    def __init__(self, dim, kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.unfold1 = nn.Conv2d(dim, dim, kernel_size, dilation=dilation[0], padding="same")
        self.unfold2 = nn.Conv2d(dim, dim, kernel_size, dilation=dilation[1], padding="same")
        self.unfold3 = nn.Conv2d(dim, dim, kernel_size, dilation=dilation[2], padding="same")
        self.conv1 = nn.Conv2d(dim * 3, dim, 3, 1, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x1 = self.unfold1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.unfold2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.unfold3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        x = self.conv1(torch.cat([x1, x2, x3], dim=1))
        x = F.gelu(x)
        x = F.dropout(x, 0.1)
        x = x.permute(0, 2, 3, 1)
        return x


class EEMoE(nn.Module):
    "Implementation of Edge-Enhanced MoE block"

    def __init__(self, dim):
        super().__init__()
        self.deconv = DEConv(dim)
        self.act1 = nn.LeakyReLU()
        self.SparseMoE = SparseMoE(dim,dim, num_experts=5, top_k=1)


    def forward(self, x):
        B, H, W, C = x.shape
        x = self.deconv(x.permute(0, 3, 1, 2))
        x = self.SparseMoE(x.permute(0,2,3,1))
        x = self.act1(x)

        return x

class MHMoE_Block(nn.Module):
    "Implementation of Morphology-aware Hierarchical MoE block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False, att_mode=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        self.norm1 = norm_layer(dim)
        self.attn = MAMoE(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop)


        self.norm2 = norm_layer(dim)
        self.dilate = MSVF(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.dec = EEMoE(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.norm4 = norm_layer(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x + self.dec(self.norm4(x))
        x_norm = self.norm1(x)
        x_att = self.attn(x_norm)
        x = x + self.drop_path(x_att)
        x = x + self.dilate(self.norm2(x))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        x = x.permute(0, 3, 1, 2)
        # B, C, H, W
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    """

    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], \
            "the patch embedding way isn't exist!"
        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 224x224
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=1, stride=1,
                          padding=0, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=1, stride=1,
                          padding=0, bias=False),  # 56x56
            )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B, C, H, W
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, C, H, W, = x.shape

        # x: B, C, H, W   expand: B, H, W, C

        x = self.expand(x.permute(0, 2, 3, 1))
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        # x: B, H, W, C
        x = self.norm(x).permute(0, 3, 1, 2)

        return x


class MHMoE_Encoder(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None, att_mode=False):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            MHMoE_Block(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block, att_mode=att_mode)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x, feat):
        for blk in self.blocks:
            x = blk(x)
        feat.append(x)
        x = self.downsample(x)
        return x


class MHMoE_Decoder(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 expand=True, merging_way=None, att_mode=False):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            MHMoE_Block(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block, att_mode=att_mode)
            for i in range(depth)])

        # patch expanding layer
        self.expand = PatchExpand2D(dim) if expand else nn.Identity()

    def forward(self, x, feat):

        x = self.expand(x) + feat
        for blk in self.blocks:
            x  = blk(x)
        return x


class MHMoE_Net(nn.Module):
    def __init__(self, img_size=448, patch_size=4, in_chans=3, num_classes=30, embed_dim=81,
                 depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], kernel_size=3, dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, norm_name="instance",
                 res_block: bool = True,
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 downsamples=[True, True, True, False],
                 expands=[True, True, True, False],
                 att_mode=[True, True, False, False],
                 cpe_per_satge=False, cpe_per_block=True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.Encoder = nn.ModuleList()
        self.Decoder = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 0 1 2 3
            encoder = MHMoE_Encoder(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                kernel_size=kernel_size,
                                dilation=dilation,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=downsamples[i_layer],
                                cpe_per_block=cpe_per_block,
                                cpe_per_satge=cpe_per_satge,
                                merging_way=merging_way,
                                att_mode=att_mode[i_layer]
                                )
            
            self.Encoder.append(encoder)

        for i_layer in range(self.num_layers - 2, -1, -1):  # 2 1 0
            decoder = MHMoE_Decoder(dim=int(embed_dim * 2 ** i_layer),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        expand=expands[i_layer],
                                        cpe_per_block=cpe_per_block,
                                        cpe_per_satge=cpe_per_satge,
                                        merging_way=merging_way,
                                        att_mode=att_mode[i_layer]
                                        )
            self.Decoder.append(decoder)

        self.final_up = Final_PatchExpand2D(dim=embed_dim, dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(embed_dim // 4, self.num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        feat = []
        i = 0
        for encoder in self.Encoder:
            x = encoder(x, feat)
            i += 1
        return x, feat

    def forward_up(self, x, feat):
        j = 0
        for decoder in self.Decoder:
            x = decoder(x, feat[-j + 2])
            j += 1
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, feat = self.forward_features(x)
        x = self.forward_up(x, feat)
        x = self.forward_final(x)

        return x


if __name__ == "__main__":
    x = torch.rand([2, 3, 448, 448]).cuda()
    m = MHMoE_Net(img_size=448, in_chans=3, embed_dim=96).cuda()
    y = m(x)

