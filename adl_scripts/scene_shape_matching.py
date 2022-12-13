import os
import os.path as osp
import json
import h5py
import open3d as o3d
import math
import numpy as np
from tqdm import tqdm
import logging, sys

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
    
    if  object_x_y_ratio - shape_x_y_ratio < threshold \
                                and \
        object_y_z_ratio - shape_y_z_ratio < threshold \
                                and \
        object_x_z_ratio - shape_x_z_ratio < threshold:
           
           return True

    return False


def search_shapenet(mesh_bb_mapping, object_name, object_size, threshold):
    res_paths = None
    mesh_paths = readData(shapenet_data_path + object_name + '/')
    res_paths = []
    for mesh_path in mesh_paths:
        bb = mesh_bb_mapping[mesh_path]
        if compare_BBs(bb.get_extent(), object_size, threshold):
            #within threshold
            res_paths.append(mesh_path)
            continue
    return res_paths


def get_sorted_args_for_shaped_matches(mesh_bb_mapping, res_paths, object_BB_size):
    bb_size_list = []
    for res_path in res_paths:
        shape_BB = mesh_bb_mapping[res_path]
        cur_bb_error = calculate_match_error(object_BB_size, shape_BB.get_extent())
        bb_size_list.append(cur_bb_error)
    return np.argsort(np.asarray(bb_size_list)), bb_size_list


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


def get_mesh_bb_mapping(shapenet_data_path, included_classes):
    mapping = {}
    for class_name in included_classes:
        obj_paths = readData(osp.join(shapenet_data_path, class_name))
        for obj_path in obj_paths:
            mapping[obj_path] = getBBFromMeshPath(obj_path)
    return mapping


#
base_path = '/home/gogebakan/workspace/Pose2Room/'
train_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/train.json'
validation_path = base_path + 'datasets/virtualhome_22_classes/splits/script_level/val.json'
# shapenet_data_path = '/home/gogebakan/workspace/pointnet_pytorch/data/myshapenet/raw_obj/'
shapenet_data_path = '/home/gogebakan/workspace/pointnet_pytorch/data/myshapenet/small_dataset/'
included_classes = ['bed', 'sofa', 'chair', 'lamp', 'table']

# Opening JSON file
def main():
    mesh_bb_mapping = get_mesh_bb_mapping(shapenet_data_path, included_classes)
    f = open(train_path)
    # returns JSON object as a dictionary
    path_list = json.load(f)
    logging.info(f'Path list size: {len(path_list)}')
    for path in tqdm(path_list):
        '''read data'''
        current_hdf = path
        logging.debug(f'Current scene: {current_hdf}')
        sample_data = h5py.File(osp.join(base_path, current_hdf), "a")
        object_nodes = get_object_nodes(sample_data)
        shape_codes = []
        # process object nodes
        for object_node in object_nodes:
            cur_size = object_node['size']
            class_name = object_node['class_name'][0].decode("utf-8")
            if class_name in included_classes:
                res_paths = search_shapenet(mesh_bb_mapping, class_name, cur_size, 5)
            
                with open('res_paths.json', 'w') as f:
                    json.dump(res_paths, f)

                #sort
                sorted_args, bb_size_list = get_sorted_args_for_shaped_matches(mesh_bb_mapping, res_paths, cur_size)
                #mesh penetration loss
                #TO-DO
                
                best_match = res_paths[sorted_args[0]]
                best_match = os.path.splitext(best_match)[0] + '.npy'
                shape_code = np.load(best_match)
                shape_codes.append(shape_code)
                logging.debug(f'Object: {class_name}\tBest match: {best_match}')
            else:
                shape_codes.append(np.zeros((1, 1024)))

        if 'shape_codes' in sample_data.keys():
            del sample_data['shape_codes']
        sample_data.create_dataset('shape_codes', data=shape_codes) 

        sample_data.close()

    
if __name__ == '__main__':
    main()


