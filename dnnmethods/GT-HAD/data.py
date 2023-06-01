from utils import ForkedPdb
import torch.utils.data as data
from block import Block_embedding

class DatasetHsi(data.Dataset):
    def __init__(self, hsi_data, wsize=15, wstride=5):
        super(DatasetHsi, self).__init__()
        self.data_processer = Block_embedding(wsize=wsize, wstride=wstride)
        self.block_gt, self.block_input, self.padding = self.data_processer(hsi_data)

    def __getitem__(self, index):
        block_gt = self.block_gt[index]
        block_input = self.block_input[index]

        return {'block_gt': block_gt, 'block_input': block_input, 'index': index}

    def __len__(self):
        return self.block_gt.size(0)