import torch

import numpy as np
import os
from tqdm import tqdm

from utility import *

class datasetsKitti(torch.utils.data.Dataset):
    def __init__(self, path, datanum:list, startidx:list =[0, 0, 0], transforms=None, progress=False):
        assert len(datanum) == 3 and len(startidx) == 3

        img_types = ['um', 'umm', 'uu']
        img_num = [95, 96, 98]
        assert datanum + startidx < img_num

        self.transforms = transforms

        basePath = os.path.join(path, 'training')
        trainPath = os.path.join(basePath, 'velodyne')

        types = np.concatenate([[t] * d for d, t in zip(datanum, img_types)])
        idx = np.concatenate([np.arange(s, s+d) for d, s in zip(datanum, startidx)])

        pbar_input = zip(types, idx)
        if progress:
            pbar_input = tqdm(list(pbar_input))
            pbar_input.set_description("Preparing input  data")
        self.image = [pcd2Cyl(bin2Pcd(os.path.join(trainPath, '%s_%06d.bin' % (t, num)))) for t, num in pbar_input]

        pbar_target = zip(types, idx, self.image)
        if progress:
            pbar_target = tqdm(list(pbar_target))
            pbar_target.set_description("Preparing target data")
        self.target = [gt2Cyl(num, path=path, img_type=t, pcd=pcd) for t, num, pcd in pbar_target]

        self.datanum = sum(datanum)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        image = self.image[idx]
        target = self.target[idx]

        if self.transforms is not None:
            image = self.transforms(image)
            target = self.transforms(target)

        return image, target