#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import os
from torch.utils.data import Dataset
from utils.tools import read_json
import h5py


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
        split_file = os.path.join(cfg.config['data']['split'], mode + '.json')
        self.split = read_json(split_file)
        self.id_file_dict = self.get_data_to_memory()


    def __len__(self):
        return len(self.split)

    def get_data_to_memory(self):
        id_file_dict = {}
        for sample_file in self.split:
            sample_data = h5py.File(sample_file, "r")
            
            obj_nodes = sample_data['object_nodes']
            id_file_dict[sample_file] = {
                "skeleton_joints" : sample_data['skeleton_joints'][:],
                "object_nodes" : {id: {'R_mat': obj_nodes[id]['R_mat'][:],
                                'centroid': obj_nodes[id]['centroid'][:],
                                'size': obj_nodes[id]['size'][:] } for id in obj_nodes.keys()},
                "shape_codes" : sample_data['shape_codes'][:]
            }
            sample_data.close()
        return id_file_dict
