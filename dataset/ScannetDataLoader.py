import os
import glob
import torch
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create, data_prepare

NUM_CLASS = 21

class ScanNet(Dataset):
    def __init__(self, args, split, coord_transform=None, rgb_transform=None,
                 rgb_mean=None, rgb_std=None, shuffle_index=False):
        super().__init__()
        self.args, self.split, self.coord_transform, self.rgb_transform, self.rgb_mean, self.rgb_std, self.shuffle_index = \
            args, split, coord_transform, rgb_transform, rgb_mean, rgb_std, shuffle_index
        self.stop_aug = False
        data_root = args.data_dir
        
        if split == "train":
            #self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
            self.data_list = sorted(os.listdir(os.path.join(data_root, split)))
            self.data_list = [item[:-4] for item in self.data_list]
            
        elif split == 'trainval':
            data_list = glob.glob(os.path.join(
                data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        elif split == 'val':        
            #self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
            self.data_list = sorted(os.listdir(os.path.join(data_root, split)))
            self.data_list = [item[:-4] for item in self.data_list]
        else:
            raise ValueError("no such split: {}".format(split))
        
        self.data_idx = np.arange(len(self.data_list))
        #print(self.data_list,self.data_idx)
        for item in self.data_list:
            if not os.path.exists("/dev/shm/scannet2_{}".format(item)):
                data_path = os.path.join(args.data_dir, split, item + '.pth')
                print(data_path)
                data = torch.load(data_path)
                #data = np.load(data_path).astype(np.float32)  # xyzrgbl, N*7
                label = np.expand_dims(data[2], axis=1)
                idx = np.where(label[:,-1] == -100)
                label[idx,-1] = 20
                con_data = np.concatenate([data[0], data[1], label], axis=1)
                #con_data = np.delete(con_data, idx, axis=0) #(81369, 7) -> (59452, 7)  
                #print(con_data.shape)
                #print(data[0].shape, data[1].shape, data[2].shape,max(con_data[:,-1]),min(con_data[:,-1]),max(data[3]),min(data[3]))
                #exit(0)
                sa_create("shm://scannet2_{}".format(item), con_data)
        
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://scannet2_{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        #print(coord.shape,max(coord[:,0]),min(coord[:,0]),max(coord[:,1]),min(coord[:,1]),max(feat[:,0]),min(feat[:,0]))
        
        coord, feat, label = \
            data_prepare(coord, feat, label, self.args, self.split, self.coord_transform, self.rgb_transform,
                         self.rgb_mean, self.rgb_std, self.shuffle_index, self.stop_aug)
        #print(coord.shape,max(coord[:,0]),min(coord[:,0]),max(coord[:,1]),min(coord[:,1]),max(feat[:,0]),min(feat[:,0]))
        #exit(0)

        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.args.loop