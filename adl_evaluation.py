#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import h5py
from adl_scripts.MLP_Regressor import MLP_Regressor
from utils.pc_utils import rot2head
import numpy as np
import open3d as o3d
import os
import os.path as osp
import torch
import json
from time import time
from tqdm import tqdm

def dist_node2bbox(nodes, joint_coordinates, joint_num):
    sk_ids = []
    for node in nodes:
        vecs = joint_coordinates - node['centroid']
        dist_offset = np.abs(vecs.dot(node['R_mat'].T)) - np.array(node['size']) / 2.
        dists = dist_offset.max(axis=-1)
        dists = np.max(dists.reshape(-1, joint_num), axis=-1)
        sk_ids.append(dists.argmin())
    return np.sort(sk_ids)


def get_even_dist_joints(skeleton_joints, skip_rates):
    # Downsampling by 1-d distance interpolation
    frame_num = skeleton_joints.shape[0] // skip_rates + 1
    movement_dist = np.linalg.norm(np.diff(skeleton_joints[:, 0], axis=0), axis=1)
    cum_dist = np.cumsum(np.hstack([[0], movement_dist]))
    target_cum_dist = np.linspace(0, sum(movement_dist), frame_num)
    selected_sk_ids = np.argmin(np.abs(cum_dist[:, np.newaxis] - target_cum_dist), axis=0)
    return selected_sk_ids


def get_mesh_shape_code_mapping(shapenet_data_path, included_classes):
    def read_data(data_path):
        shape_code_paths = []
        # giving file extension
        ext = ('.npy')
        for file_name in os.listdir(data_path):
            if file_name.endswith(ext):
                shape_code_paths.append(osp.join(data_path, file_name))
        return shape_code_paths
    
    mapping = {}
    for class_name in included_classes:
        shape_code_paths = read_data(osp.join(shapenet_data_path, class_name))
        mapping[class_name] = []
        for shape_code_path in shape_code_paths:
            shape_code = np.load(shape_code_path).flatten()
            shape_code = torch.from_numpy(shape_code).cuda()
            mapping[class_name].append((shape_code_path, shape_code))
    return mapping
            
            
def find_closest_shapenet_model(output_shape_code, obj_class, mesh_shape_code_mapping):
    min_dist = 10**9
    shape_code_path = None

    for shape_path, shape_code in mesh_shape_code_mapping[obj_class]:
        dist = torch.linalg.norm(shape_code - output_shape_code)
        if dist < min_dist:
            min_dist = dist
            shape_code_path = shape_path
    
    obj_path = shape_code_path.replace('.npy', '.obj')
    return obj_path

def mesh_penetration_loss(mesh_path, nearest_k_frames, weight=0.8):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

    total_loss = 0.0
    
    for frame in nearest_k_frames:
        frame = frame.astype('float32')
        signed_distances = scene.compute_signed_distance(frame).numpy()
        negative_indices = signed_distances > 0
        signed_distances = signed_distances**2
        signed_distances[negative_indices] *= weight
        total_loss += signed_distances.sum()

    return total_loss

def create_adl_input(sample_data):
    node_data = sample_data['object_nodes'][str(idx)]
    class_name = node_data['class_name'][0].decode('utf-8')

    nearest_seed_skeleton_feature = sample_data['nearest_seed_skeleton_features'][idx].flatten()
    heading = rot2head(node_data['R_mat'])
    box3D = np.hstack([node_data['centroid'], np.log(node_data['size']), np.sin(heading), np.cos(heading)])
    adl_input = np.hstack([box3D, nearest_seed_skeleton_feature]).astype(np.float32)
    
    return adl_input, class_name

def get_nearest_k_frames(sample_data):
    skeleton_joints = sample_data['skeleton_joints'][:]
    nearest_k_frames_indices = sample_data['nearest_seed_skeleton_indices'][idx].astype('int')
    input_joints_indices = np.linspace(0, skeleton_joints.shape[0]-1, 768).round().astype(np.uint16)
    input_joints = skeleton_joints[input_joints_indices]
    nearest_k_frames = input_joints[nearest_k_frames_indices]

    return nearest_k_frames

def get_mlp_model(train_params):
    checkpoint_path = train_params['checkpoint_path']
    input_size = train_params['input_size']
    layer_sizes = train_params['layer_sizes']
    output_size = train_params['output_size']
    model = MLP_Regressor(input_size=input_size, output_size=output_size, layer_sizes=layer_sizes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
    model.eval()

    return model 


if __name__ == '__main__':

    base_path = '/home/baykara/adl4cv/Pose2Room/'
    # test_json_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/small.json'
    test_json_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/test.json'

    with open(test_json_path) as f:
        test_scenes = json.load(f)

    with open('configs/config_files/train_params.json') as f:
        train_params = json.load(f)
    
    model = get_mlp_model(train_params)

    checkpoint_name = train_params["checkpoint_path"].split('/')[-1]
    print(f'model with {checkpoint_name} is loaded')

    shapenet_data_path = train_params['shapenet_data_path']
    included_classes = train_params['included_classes']
    mesh_shape_code_mapping = get_mesh_shape_code_mapping(shapenet_data_path, included_classes)

    total_mesh_penetration_loss = 0.0
    start = time()
    for scene_path in tqdm(test_scenes):
        sample_data = h5py.File(scene_path, "r")

        for idx in range(len(sample_data['object_nodes'])):
            adl_input, class_name = create_adl_input(sample_data)
            adl_input = torch.tensor(adl_input).cuda()
            output_shape_code = model(adl_input)

            best_shapenet_model_path = find_closest_shapenet_model(output_shape_code, class_name, mesh_shape_code_mapping)
            nearest_k_frames = get_nearest_k_frames(sample_data)
            total_mesh_penetration_loss += mesh_penetration_loss(best_shapenet_model_path, nearest_k_frames, weight=0.8)
        
        sample_data.close()

    end = time()
    print(f'total mesh penetration loss: {total_mesh_penetration_loss:.2f}')
    print(f'average mesh penetration loss: {(total_mesh_penetration_loss/len(test_scenes)):.2f}')
    print(f'average time: {(end-start)/len(test_scenes):.2f}')

'''
model with checkpoint3.pt is loaded
average mesh penetration loss: 40809.47

model with mpl_checkpoint3.pt is loaded
average mesh penetration loss: 38527.15
'''