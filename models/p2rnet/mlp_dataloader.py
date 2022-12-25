#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from models.datasets import Base_Dataset
import h5py
import os
from utils.pc_utils import rot2head
import random

default_collate = torch.utils.data.dataloader.default_collate


class MLP_Dataset(Base_Dataset):
    def __init__(self,cfg, mode):
        super(MLP_Dataset, self).__init__(cfg, mode)
        self.num_frames = cfg.config['data']['num_frames']
        self.use_height = not cfg.config['data']['no_height']
        self.max_num_obj = cfg.config['data']['max_gt_boxes']

    def __getitem__(self, idx):
        '''Get each sample'''
        '''Load data'''
        sample_data = self.data_list[idx]
        object_node = sample_data['object_node']
        shape_code = sample_data['shape_code']
        nearest_seed_skeleton_feature = sample_data['nearest_seed_skeleton_feature'].flatten()

        heading = rot2head(object_node['R_mat'])
        box3D = np.hstack([object_node['centroid'], np.log(object_node['size']), np.sin(heading), np.cos(heading)])

        # deliver to network
        ret_dict = {}
        ret_dict['adl_input'] = np.hstack([box3D, nearest_seed_skeleton_feature]).astype(np.float32)
        ret_dict['adl_output'] = shape_code.astype(np.float32)
        
        return ret_dict

def collate_fn(batch):
    '''
    data collater
    :param batch:
    :return:
    '''
    collated_batch = {}
    for key in batch[0]:
        if key not in ['sample_idx']:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        else:
            collated_batch[key] = [elem[key] for elem in batch]
    return collated_batch

class Custom_Dataloader(object):
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Init datasets and dataloaders
def MLP_dataloader(cfg, mode='train'):
    if cfg.config['data']['dataset'] == 'virtualhome':
        dataset = MLP_Dataset(cfg, mode)
    else:
        raise NotImplementedError

    if cfg.config['device']['distributed']:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
    else:
        if mode=='train':
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=cfg.config[mode]['batch_size'],
                                                  drop_last=False)

    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            num_workers=cfg.config['device']['num_workers'],
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    dataloader = Custom_Dataloader(dataloader, sampler)
    return dataloader
