#  P2RNet: model loader
#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
import numpy as np
import os.path as osp
import h5py

@METHODS.register_module
class P2RNet(BaseNetwork):
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        phase_names = []
        if cfg.config[cfg.config['mode']]['phase'] in ['full']:
            phase_names += ['backbone', 'centervoting', 'detection']

        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1), cfg.config['device']['gpu'], cfg))

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def generate(self, data, eval=True):
        '''
        Forward pass of the network for object detection
        '''
        # --------- Backbone ---------
        end_points = {}
        end_points = self.backbone(data['input_joints'], end_points)
        xyz = end_points['seed_skeleton']
        features = end_points['seed_features']
        seed_inds = end_points['seed_inds']


        #######
        def get_nearest_K_features(seed_features, seed_skeletons, seed_inds, BB_center, K = 10):
            root_joints = seed_skeletons[0,:,0,:]
            dist = torch.norm(root_joints - BB_center, dim=1, p=None)
            min_seed_index = torch.argmin(dist)
            if min_seed_index < K:
                min_seed_index = K
            elif min_seed_index > seed_features.shape[1] - K:
                min_seed_index = seed_features.shape[1] - K
            knn_seed_features = seed_features[0][min_seed_index-K:min_seed_index+K]
            knn_seed_inds = seed_inds[0][min_seed_index-K:min_seed_index+K]
            return knn_seed_features.cpu().detach().numpy(), knn_seed_inds.cpu().detach().numpy()

        def write_nearest_features_to_dataset(file_name, samples_folder_path, nearest_seed_skeleton_features, nearest_seed_skeleton_indices):
            file_dest = osp.join(samples_folder_path, file_name)
            sample_data = h5py.File(file_dest, "a")
            if 'nearest_seed_skeleton_features' in sample_data.keys():
                del sample_data['nearest_seed_skeleton_features']
            if 'nearest_seed_skeleton_indices' in sample_data.keys():
                del sample_data['nearest_seed_skeleton_indices']
            sample_data.create_dataset('nearest_seed_skeleton_features', data=nearest_seed_skeleton_features) 
            sample_data.create_dataset('nearest_seed_skeleton_indices', data=nearest_seed_skeleton_indices) 
            sample_data.close()


        K = 10

        nearest_seed_skeleton_features_list = []
        nearest_seed_skeleton_indices_list = []
        for BB_center in data['center_label'][0]:
            if torch.count_nonzero(BB_center) != 0:
                knn_seed_features, knn_seed_inds = get_nearest_K_features(features, xyz, seed_inds, BB_center, K)
            else:
                knn_seed_features = np.zeros((2*K, features.shape[-1]))
                knn_seed_inds = np.zeros(2*K)
            nearest_seed_skeleton_features_list.append(knn_seed_features)
            nearest_seed_skeleton_indices_list.append(knn_seed_inds)
          
        samples_folder_path = self.cfg.config['data']['samples_path']
        file_name = f"{data['sample_idx'][0]}.hdf5"
        # try:
        write_nearest_features_to_dataset(file_name, 
                                        samples_folder_path, 
                                        nearest_seed_skeleton_features_list, 
                                        nearest_seed_skeleton_indices_list)
        # except:
        #     breakpoint()
        #######

        # --------- Generate Center Candidates ---------
        xyz, features = self.centervoting(xyz, features)
        features_norm = torch.norm(features, p=2, dim=2)
        features = features.div(features_norm.unsqueeze(2))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # --------- DETECTION ---------
        end_points, _ = self.detection.generate(xyz, features, end_points, False)
        eval_dict, parsed_predictions = parse_predictions(end_points, data, self.cfg.eval_config)
        eval_dict = assembly_pred_map_cls(eval_dict, parsed_predictions, self.cfg.eval_config)

        if eval:
            '''Get meta data for evaluation'''
            parsed_gts = parse_groundtruths(data, self.cfg.eval_config)

            '''for mAP evaluation'''
            eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts)

        return end_points, eval_dict, parsed_predictions

    def forward(self, data):
        '''
        Forward pass of the network
        :param data (dict): contains the data for training.
        :return: end_points: dict
        '''
        # --------- Backbone ---------
        end_points = {}
        end_points = self.backbone(data['input_joints'], end_points)
        xyz = end_points['seed_skeleton']
        features = end_points['seed_features']

        # --------- Generate Center Candidates ---------
        xyz, features = self.centervoting(xyz, features)
        features_norm = torch.norm(features, p=2, dim=2)
        features = features.div(features_norm.unsqueeze(2))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # --------- DETECTION ---------
        end_points, _ = self.detection(xyz, features, end_points, False)
        return end_points

    def loss(self, pred_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        if isinstance(pred_data, tuple):
            pred_data = pred_data[0]

        total_loss = self.detection_loss(pred_data, gt_data, self.cfg.dataset_config)
        return total_loss
