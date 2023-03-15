import torch
import torch.nn.functional as F
import torch.nn as nn 
import pdb


class Patch_embedding(nn.Module):
    def __init__(self, ksize_outer=15, stride_outer=5):
        super(Patch_embedding, self).__init__()
        self.ksize = ksize_outer
        self.stride = stride_outer

    def same_padding(self, images, ksizes, strides, rates=(1, 1)):
        assert len(images.size()) == 4
        batch_size, channel, rows, cols = images.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        images = F.pad(images, paddings, mode='replicate') # replicate reflect 

        return images, paddings

    def extract_image_patches(self, images, ksizes, strides):
        assert len(images.size()) == 4
        images, paddings = self.same_padding(images, ksizes, strides)
        unfold = torch.nn.Unfold(kernel_size=ksizes,
                                 padding=0,
                                 stride=strides)
        patches = unfold(images)

        return patches, paddings

    def forward(self, x):
        _, band, row, col = x.size()
        
        # patch oder: from left to right and from top to down
        patch_outer, paddings = self.extract_image_patches(x, ksizes=[self.ksize, self.ksize],
                    strides=[self.stride, self.stride],) # batch, c*h*w, num
        patch_outer = patch_outer.squeeze(0).permute(1, 0) # num1, l=c*h*w
        num1 = patch_outer.size(0)
        patch_outer = patch_outer.view(num1, band, self.ksize, self.ksize)

        return patch_outer, patch_outer, paddings


class Patch_fold(nn.Module):
    def __init__(self, ksize_outer=15, stride_outer=5):
        super(Patch_fold, self).__init__()
        self.ksize = ksize_outer
        self.stride = stride_outer

    def forward(self, x, paddings, row, col):
        num = x.size(0)
        back = x.view(num, -1).permute(1, 0).unsqueeze(0)

        # 判断padding
        block_size1 = (row + 2 * paddings[0] - (self.ksize - 1) - 1) / self.stride + 1
        block_size2 = (col + 2 * paddings[1] - (self.ksize - 1) - 1) / self.stride + 1
        if block_size1 * block_size2 != num:
            pad = [paddings[1], paddings[3]]
        else:
            pad = [paddings[0], paddings[2]]

        try:
            ori = F.fold(back, (row, col), (self.ksize, self.ksize), padding=pad, stride=self.stride)
        except:
            pdb.set_trace()
        tmp = torch.ones_like(ori)
        tmp_unfold = F.unfold(tmp, (self.ksize, self.ksize), padding=pad, stride=self.stride)
        fold_mask = F.fold(tmp_unfold, (row, col), (self.ksize, self.ksize), padding=pad, stride=self.stride)
        out = ori / fold_mask

        return out


class Patch_search(nn.Module):
    def __init__(self, x_ori, gt_var, ksize_key=15, stride_key=5):
        super(Patch_search, self).__init__()
        self.ksize = ksize_key
        self.stride = stride_key
        self.dis = torch.nn.PairwiseDistance(p=2, keepdim=True)

        patch_key, _ = self.extract_image_patches(x_ori, ksizes=[self.ksize, self.ksize],
                    strides=[self.stride, self.stride])
        self.patch_key = patch_key.squeeze(0).permute(1, 0) # num2, l=c*h*w

        ab_gt, _ = self.extract_image_patches(gt_var, ksizes=[self.ksize, self.ksize],
                    strides=[self.stride, self.stride])
        self.ab_gt = ab_gt.squeeze(0).permute(1, 0) # num2, l=c*h*w

    def same_padding(self, images, ksizes, strides, rates=(1, 1)):
        assert len(images.size()) == 4
        batch_size, channel, rows, cols = images.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
        # Pad the input
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        images = F.pad(images, paddings, mode='replicate') # replicate reflect

        return images, paddings

    def extract_image_patches(self, images, ksizes, strides):
        assert len(images.size()) == 4
        images, paddings = self.same_padding(images, ksizes, strides)
        unfold = torch.nn.Unfold(kernel_size=ksizes,
                                 padding=0,
                                 stride=strides)
        patches = unfold(images)

        return patches, paddings

    def matrix_get_dis(self, x1, x2):
        b = x1.size(0)
        items = [self.dis(x1[i].unsqueeze(0), x2).unsqueeze(0) for i in range(b)]
        dis_map = torch.cat(items, dim=0).squeeze(-1)

        return dis_map

    def forward(self, x_out, match_flag, idx):
        patch_query, _ = self.extract_image_patches(x_out, ksizes=[self.ksize, self.ksize],
                    strides=[self.stride, self.stride])
        patch_query = patch_query.squeeze(0).permute(1, 0) # num2, l=c*h*w
        num = patch_query.size(0)

        dis_map = self.matrix_get_dis(patch_query, self.patch_key)
        _, index = torch.topk(dis_map, 1, dim=1, largest=False, sorted=True)
        index = index.squeeze()

        flag = (index == idx).int()
        match_flag[flag == 1] = 1

        return match_flag

