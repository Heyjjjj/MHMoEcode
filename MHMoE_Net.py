#mlp expert
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg
from DilatedStripConv import DynamicConv, DiletedConv
from moe import SparseMoE
from MA_MoE import Adapter_MoElayer, MLP_MoElayer
from EE_MoE import De_MoElayer
from einops import rearrange, repeat
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


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
        # 初始化 fc1 的权重和偏置
        init.normal_(self.fc1.weight, std=0.001)
        init.constant_(self.fc1.bias, 0)
        # 初始化 fc2 的权重和偏置
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
        # # print(C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.head_dim,
        #                           C // self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MAMoE(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=1, extend_scope=1, if_offset=True, device='cuda'):
        super().__init__()
        #ds_inputsize = int(dim/96*112/(dim/96)*112/(dim/96))+
        self.dim = dim
        self.head_dim = dim // 3
        #print('head_dim', head_dim)
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        

        self.expert1 = nn.Conv2d(self.head_dim, self.head_dim, 3, padding=1, bias=True)
        self.expert2 = nn.Conv2d(self.head_dim, self.head_dim, 3, padding=2, dilation=2, bias=True)
        self.expert3 = DynamicConv(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=0,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.expert4 = DiletedConv(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=0,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.expert5 = DynamicConv(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=1,
            if_offset=self.if_offset,
            device='cuda',
        )
        self.expert6 = DiletedConv(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=self.kernel_size * self.kernel_size,
            extend_scope=self.extend_scope,
            morph=1,
            if_offset=self.if_offset,
            device='cuda',
        )
        
        self.moe_layers = nn.ModuleList([
            Adapter_MoElayer(
                dim=self.head_dim,
                num_experts=2,
                noisy_gating=True,
                k=2,
                expert_layers=[self.expert1, self.expert2],
                expert_type='conv'
            ),
            Adapter_MoElayer(
                dim=self.head_dim,
                num_experts=2,
                noisy_gating=True,
                k=2,
                expert_layers=[self.expert3, self.expert4],
                expert_type='conv'
            ),
            Adapter_MoElayer(
                dim=self.head_dim,
                num_experts=2,
                noisy_gating=True,
                k=2,
                expert_layers=[self.expert5, self.expert6],
                expert_type='conv'
            )
        ])
        
        heads_per_branch = max(1, num_heads // 3)
        self.attentions = nn.ModuleList([
            GlobalAttention(self.head_dim, heads_per_branch, qkv_bias, qk_scale, attn_drop)
            for _ in range(3)
        ])
        
        self.qkv_projs = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim * 3, bias=qkv_bias)
            for _ in range(3)
        ])
        
        self.final_moe = MLP_MoElayer(
            dim=dim,
            num_experts=3,
            noisy_gating=True,
            k=2,
            mlp_ratio=4
        )


        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        

    def forward(self, x):
        B, H, W, C = x.shape
        if torch.any(torch.isnan(x)):
            print('forward',x.size())
            
        x = rearrange(x, 'b h w (n c) -> n b h w c', n=3, c=C//3)  # [3, B, H, W, C//3]
        
        branch_outputs = []
        total_loss = 0
        
        for i in range(3):
            branch_input = x[i].permute(0, 3, 1, 2)  # [B, C//3, H, W]
            
            moe_output, loss = self.moe_layers[i](branch_input.permute(0, 2, 3, 1))
            moe_output = F.dropout(moe_output, p=0.1, training=self.training)
            total_loss += loss
            
            qkv = self.qkv_projs[i](moe_output)
            qkv = rearrange(qkv, 'b h w (n d) -> n b h w d', n=3, d=self.head_dim)  # [3, B, H, W, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn_output = self.attentions[i](q, k, v)  # [B, H, W, C//3]
            
            branch_outputs.append(attn_output)
        

        x = torch.cat(branch_outputs, dim=-1)
        
        x, final_loss = self.final_moe(x)
        x = F.dropout(x, p=0.1, training=self.training) 
        total_loss += final_loss
        
        x = self.proj(x)
        x = self.proj_drop(x)
        

        return x


class MSVF(nn.Module):
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
    def __init__(self, dim):
        super().__init__()
        self.act1 = nn.LeakyReLU()
        self.demoe = De_MoElayer(dim, num_experts=5, top_k=2)


    def forward(self, x):
        B, H, W, C = x.shape
        x = self.demoe(x)
        x = self.act1(x)
        x = F.dropout(x, p=0.1, training=self.training)

        return x

class MHMoE_Block(nn.Module):
    "Implementation of Dilate-attention block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False, att_mode=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation

        
        self.norm1 = norm_layer(dim)
        self.mamoe = MAMoE(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        self.msvf = MSVF(dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.eemoe = EEMoE(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.norm4 = norm_layer(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x + self.eemoe(self.norm1(x))
        x = x + self.mamoe(self.norm2(x))
        x = x + self.msvf(self.norm3(x))
        x = x + self.drop_path(self.mlp(self.norm4(x)))
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
        #print(x.shape)
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
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
    # x1 = torch.rand([1,3,448,448]).cuda()
    # x1 = x + x1
    # print(x1.size())
    m = Dilateformer(img_size=448, in_chans=3, embed_dim=96).cuda()
    # x = torch.randn(3,448,448).cuda()
    #stat(m,(3,448,448))
    y = m(x)
    # criterion = nn.BCEWithLogitsLoss()
    # loss = criterion(y, torch.ones([1, 30, 448, 448]).cuda())
    # loss.backward()
    #print(y.shape)
