#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import os
from torch.utils.data import Dataset
from utils.tools import read_json
import h5py
import numpy as np

class Base_Dataset(Dataset):
    def __init__(self, cfg, mode):
        '''
        initiate a base dataset for data loading in other networks
        :param cfg: config file
        :param mode: train/val/test mode
        '''
        self.config = cfg.config
        self.dataset_config = cfg.dataset_config
        self.mode = mode
        
        json_file = 'small.json' if 'use_all' in cfg.config['data'] else mode + '.json'
        # json_file = f'{mode}_all.json' if 'use_all' in cfg.config['data'] else mode + '.json'
        split_file = os.path.join(cfg.config['data']['split'], json_file)
        self.split = read_json(split_file)
        self.data_list = self.get_data_to_memory()


    def __len__(self):
        return len(self.split)

    def get_data_to_memory(self):
        data_list = []
        for sample_file in self.split:
            sample_data = h5py.File(sample_file, "r")
            obj_nodes = sample_data['object_nodes']
            nearest_seed_skeleton_features = sample_data['nearest_seed_skeleton_features'][:]
            shape_codes = sample_data['shape_codes'][:]

            for obj_ind, nearest_seed_skeleton_feature, shape_code in zip(obj_nodes, nearest_seed_skeleton_features, shape_codes):
                obj_node = obj_nodes[obj_ind]
                if np.count_nonzero(obj_node['size'][:]) != 0:
                    data = {
                        "nearest_seed_skeleton_feature" : nearest_seed_skeleton_feature,
                        "object_node" : {'R_mat': obj_node['R_mat'][:],
                                        'centroid': obj_node['centroid'][:],
                                        'size': obj_node['size'][:]},
                        "shape_code" : shape_code.flatten()
                    }
                    data_list.append(data)
            sample_data.close()
        return data_list
