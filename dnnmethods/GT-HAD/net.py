import torch.nn as nn
import torch
import pdb

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, 
                out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, patch_size=3, patch_stride=3, attn_drop=0.):
        super().__init__()
        self.psize = patch_size
        self.pstride = patch_stride  # Wh, Ww

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.embed_dim = dim 
        self.hidden_dim = self.embed_dim // 2
        self.fc = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.scale = (self.hidden_dim * self.psize ** 2) ** -0.5
        # creat a mask for the calculations of AFB and BFB
        self.N = self.psize * self.pstride # patch num
        self.mask = torch.eye(self.N).cuda()
        self.mask[self.mask == 1] = -100.0

    def calculate_mask(self, block_idx=0, match_vec=None):
        B = block_idx.size(0)
        # jduge which blocks are searched
        cur_match = torch.index_select(match_vec, dim=0, index=block_idx).squeeze()

        # all blocks undergo AFB
        if cur_match.sum() == 0:
            mask = self.mask
        else:
            mask = self.mask.unsqueeze(0).repeat(B, 1, 1)
            # searched blocks undergo BFB
            mask[cur_match == 1] = 0

        return mask

    def attn_cal(self, attn, mask, v, shape):
        B, H, W, C = shape
        attn = attn + mask
        attn = self.softmax(attn)

        x_attn = (attn @ v)
        x_back = x_attn.view(B, self.pstride, self.pstride, self.psize, self.psize, C)
        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        return x

    def forward(self, x, block_idx=0, match_vec=None):
        B, H, W, C = x.shape
        N = self.N
        P = self.psize ** 2

        # attention calculation
        x_view = x.view(B, self.pstride, self.psize, self.pstride, self.psize, C)
        x_fc = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, P, C) 
        x_q = self.fc(x_fc).view(B, N, -1)
        attn = x_q @ x_q.transpose(-2, -1)
        attn = attn * self.scale
        attn = self.attn_drop(attn)

        v = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, P * C)
        mask = self.calculate_mask(block_idx=block_idx, match_vec=match_vec)
        x = self.attn_cal(attn, mask, v, [B, H, W, C])

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, patch_size=3, patch_stride=3, mlp_ratio=4., 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop=0., drop=0.):
        super().__init__()

        # GTB: GDBN
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, patch_size=patch_size, patch_stride=patch_stride, attn_drop=attn_drop)
        # GTB: FFN
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, block_idx=0, match_vec=None):
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.norm1(x)
        # GDBN
        x = x.view(B, H, W, C)
        x = self.attn(x, block_idx=block_idx, match_vec=match_vec)
        # FFN
        x = x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, H, W, C)

        return x

class Net(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=3, patch_stride=3, 
            mlp_ratio=2., attn_drop=0., drop=0.):
        super(Net, self).__init__()
        # head
        self.conv_head = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        # attn_layer
        self.attn_layer = TransformerBlock(embed_dim, patch_size=patch_size, 
            patch_stride=patch_stride, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop)
        # tail
        self.conv_tail = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x, block_idx=0, match_vec=None):
        x = self.conv_head(x) # b, c, h, w
        x = x.permute(0, 2, 3, 1).contiguous() # b, h, w, c
        x = self.attn_layer(x, block_idx=block_idx, match_vec=match_vec)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_tail(x)

        return x
