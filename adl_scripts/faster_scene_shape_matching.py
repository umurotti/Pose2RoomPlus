import os
import os.path as osp
import json
import h5py
import open3d as o3d
import math
import numpy as np
from tqdm import tqdm
import logging, sys
import pickle
import heapq

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def get_object_nodes(sample_data):
    object_nodes = []
    for idx in range(len(sample_data['object_nodes'])):
        object_node = {}
        node_data = sample_data['object_nodes'][str(idx)]
        for key in node_data.keys():
            if node_data[key].shape is None:
                continue
            object_node[key] = node_data[key][:]
        object_nodes.append(object_node)
    return object_nodes

def readData(dataPath):
    meshPaths = []
    # giving file extension
    ext = ('.obj')
    for file_name in os.listdir(dataPath):
        if file_name.endswith(ext):
            meshPaths.append(osp.join(dataPath, file_name))
    return meshPaths

def getBBFromMeshPath(meshPath):
    mesh = o3d.io.read_triangle_mesh(meshPath)
    pcd = mesh.sample_points_uniformly(number_of_points=1024)
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    return bb

def compare_BBs(shape_BB_size, object_BB_size, threshold):
    object_x_y_ratio = object_BB_size[0] / object_BB_size[1]
    object_y_z_ratio = object_BB_size[1] / object_BB_size[2]
    object_x_z_ratio = object_BB_size[0] / object_BB_size[2]
    
    shape_x_y_ratio = shape_BB_size[0] / shape_BB_size[1]
    shape_y_z_ratio = shape_BB_size[1] / shape_BB_size[2]
    shape_x_z_ratio = shape_BB_size[0] / shape_BB_size[2]
    
    if  abs(object_x_y_ratio - shape_x_y_ratio) < threshold and \
        abs(object_y_z_ratio - shape_y_z_ratio) < threshold and \
        abs(object_x_z_ratio - shape_x_z_ratio) < threshold:   
        return True

    return False

def search_shapenet(mesh_bb_mapping, object_name, object_size, threshold):
    mesh_paths = readData(osp.join(shapenet_data_path, object_name))
    res_paths = []
    for mesh_path in mesh_paths:
        bb = mesh_bb_mapping[mesh_path]
        if compare_BBs(bb, object_size, threshold):
            #within threshold
            res_paths.append(mesh_path)
            
    return np.array(res_paths)

def calculate_match_error(object_BB_size, shape_BB_size):
    object_x_y_ratio = object_BB_size[0] / object_BB_size[1]
    object_y_z_ratio = object_BB_size[1] / object_BB_size[2]
    object_x_z_ratio = object_BB_size[0] / object_BB_size[2]
    
    shape_x_y_ratio = shape_BB_size[0] / shape_BB_size[1]
    shape_y_z_ratio = shape_BB_size[1] / shape_BB_size[2]
    shape_x_z_ratio = shape_BB_size[0] / shape_BB_size[2]
    error = (object_x_y_ratio - shape_x_y_ratio)**2 + \
            (object_y_z_ratio - shape_y_z_ratio)**2 + \
            (object_x_z_ratio - shape_x_z_ratio)**2
    return math.sqrt(error)

def get_sorted_args_for_shaped_matches(mesh_bb_mapping, res_paths, object_BB_size):
    bb_size_list = []
    for res_path in res_paths:
        shape_BB = mesh_bb_mapping[res_path]
        cur_bb_error = calculate_match_error(object_BB_size, shape_BB)
        bb_size_list.append(cur_bb_error)
    return np.argsort(np.asarray(bb_size_list)), bb_size_list

def get_top_k_paths_from_shapenet(mesh_bb_mapping, res_paths, object_BB_size, k=10):
    bb_errors = []
    max_error_tuple = (-1, '')

    for res_path in res_paths:
        shape_BB = mesh_bb_mapping[res_path]
        cur_bb_error = calculate_match_error(object_BB_size, shape_BB)
        if len(bb_errors) <= k:
            heapq.heappush(bb_errors, (cur_bb_error, res_path))
            if cur_bb_error > max_error_tuple[0]:
                max_error_tuple = (cur_bb_error, res_path)

        elif cur_bb_error < max_error_tuple[0]:
            bb_errors.remove(max_error_tuple)
            max_error_tuple = (cur_bb_error, res_path)
            heapq.heapify(bb_errors)
            heapq.heappush(bb_errors, max_error_tuple)

    return bb_errors
    # return [path for (err, path) in bb_errors]

def get_mesh_bb_mapping(shapenet_data_path, included_classes):
    mapping = {}
    for class_name in included_classes:
        obj_paths = readData(osp.join(shapenet_data_path, class_name))
        for obj_path in obj_paths:
            mapping[obj_path] = getBBFromMeshPath(obj_path)
    return mapping

def mesh_penetration_loss(mesh_path, nearest_k_frames, weight=0.8):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

    nearest_k_frames = nearest_k_frames.reshape(-1, 3).astype('float32')
    signed_distances = scene.compute_signed_distance(nearest_k_frames).numpy()
    positive_indices = signed_distances > 0
    signed_distances = signed_distances**2
    signed_distances[positive_indices] *= weight
    total_loss = signed_distances.sum()

    return total_loss

user = 'baykara'
if user == 'gogebakan':
    base_path = '/home/gogebakan/workspace/Pose2Room/'
    train_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/train.json'
    validation_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/val.json'
    # shapenet_data_path = '/home/gogebakan/workspace/pointnet_pytorch/data/myshapenet/raw_obj/'
    shapenet_data_path = '/home/gogebakan/workspace/pointnet_pytorch/data/myshapenet/small_dataset/'
else:
    base_path = '/home/baykara/adl4cv/Pose2Room/'
    train_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/train_fast.json'
    validation_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/val_fast.json'
    shapenet_data_path = '/home/baykara/adl4cv/pointnet_pytorch/data/adl_shapenet/watertight'
    
included_classes = ['bench', 'cabinet', 'faucet', 'stove', 'bookshelf', 'computer', 'desk', 'chair', 'monitor', 'sofa', 'lamp', 'nightstand', 'bed', 'dishwasher', 'fridge', 'microwave', 'toilet']

class_thresholds = {
    'bench': 1.0,
    'cabinet': 1.35,
    'faucet': 0.15,
    'stove': 0.2,
    'bookshelf': 0.8,
    'computer': 1.8,
    'desk': 0.6,
    'chair': 1.6,
    'monitor': 2.0, # min 10
    'sofa': 0.4, # min 14
    'lamp': 2.6, # min 16 
    'nightstand': 0.6, # min 20
    'bed': 1.5, # min 16
    'dishwasher': 0.25, # min 24
    'fridge': 10.0, # min 11
    'microwave': 0.6, # min 18
    'toilet': 1.0 # min 10
    }

with open('mesh_bb_mapping', 'rb') as f:
    mesh_bb_mapping = pickle.load(f)

for json_path in [train_path, validation_path]:
    with open(json_path) as f:
        # returns JSON object as a dictionary
        path_list = json.load(f)
    
    for current_hdf in tqdm(path_list):
    # for current_hdf in tqdm(path_list):
        logging.debug(f'Current scene: {current_hdf}')
        sample_data = h5py.File(osp.join(base_path, current_hdf), "a")
        object_nodes = get_object_nodes(sample_data)
        shape_codes = []
        # process object nodes
        for i,object_node in enumerate(object_nodes):
            cur_size = object_node['size']
            class_name = object_node['class_name'][0].decode("utf-8")
            if class_name in included_classes:
            # if class_name == 'chair':
                # res_paths = search_shapenet(mesh_bb_mapping, class_name, cur_size, 2.5)
                res_paths = search_shapenet(mesh_bb_mapping, class_name, cur_size, class_thresholds[class_name])
                new_threshold = class_thresholds[class_name]
                while len(res_paths) == 0:
                    new_threshold += 0.5
                    res_paths = search_shapenet(mesh_bb_mapping, class_name, cur_size, new_threshold)

                # print(f'{current_hdf} {class_name} {len(res_paths)}')
                
                #sort
                sorted_args, bb_size_list = get_sorted_args_for_shaped_matches(mesh_bb_mapping, res_paths, cur_size)
                top10_mesh_indices = sorted_args[:10]
                top10_mesh_paths2 = res_paths[top10_mesh_indices]
                tmp = [(path, error) for path, error in zip(top10_mesh_paths2, bb_size_list)]

                top10_mesh_paths = get_top_k_paths_from_shapenet(mesh_bb_mapping, res_paths, cur_size, k=10)
                breakpoint()
                #mesh penetration loss
                nearest_k_frames_indices = sample_data['nearest_seed_skeleton_indices'][i].astype('int')
                skeleton_joints = sample_data['skeleton_joints'][:]
                input_joints_indices = np.linspace(0, skeleton_joints.shape[0]-1, 768).round().astype(np.uint16)
                input_joints = skeleton_joints[input_joints_indices]
                nearest_k_frames = input_joints[nearest_k_frames_indices]

                mesh_penetration_losses = [mesh_penetration_loss(curr_mesh_path, nearest_k_frames) for curr_mesh_path in top10_mesh_paths]
                best_mesh_index = np.argmin(mesh_penetration_losses)
                
                best_match = top10_mesh_paths[best_mesh_index]
                best_match = os.path.splitext(best_match)[0] + '.npy'
                shape_code = np.load(best_match)
                shape_codes.append(shape_code)
                logging.debug(f'Object: {class_name}\tBest match: {best_match}')
            else:
                shape_codes.append(np.zeros((1, 1024)))

        # shape codes for each scene have to be in size (10, 1024)
        for i in range(len(shape_codes), 10):
            shape_codes.append(np.zeros((1, 1024)))
        if 'shape_codes' in sample_data.keys():
            del sample_data['shape_codes']
        sample_data.create_dataset('shape_codes', data=shape_codes) 
        sample_data.close()
        









