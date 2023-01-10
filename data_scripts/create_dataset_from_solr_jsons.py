import json
import os
import os.path as osp
import shutil
from pathlib import Path
from tqdm import tqdm


labels = ['bench', 'cabinet', 'faucet', 'stove', 'bookshelf', 'computer', 'desk', 'chair', 'monitor', 'sofa', 'lamp', 'nightstand', 'bed', 'dishwasher', 'fridge', 'microwave', 'toilet']
jsons_dir = '/home/baykara/adl4cv/pointnet_pytorch/data/solr_jsons'
out_data_dir = '/home/baykara/adl4cv/pointnet_pytorch/data/adl_shapenet/watertight'
org_shapenet_dir = '/home/baykara/adl4cv/pointnet_pytorch/data/shapenetv2'

# for label in tqdm(labels):
#     file_path = osp.join(jsons_dir, f'{label}.json')
#     with open(file_path, 'r') as f:
#         json_file = json.load(f)
#         instances = json_file['response']['docs']

#         # create a folder for the current class
#         class_path = osp.join(out_data_dir, label)
#         Path(class_path).mkdir(parents=True, exist_ok=True)

#         for i,instance in enumerate(instances):
#             # synsetId = instance['synsetId']
#             # fullId = instance['fullId'][0].split('.')[-1]
#             # instance_path = osp.join(org_shapenet_dir, synsetId, fullId, 'models/model_normalized.obj')
#             # out_file_path = osp.join(class_path, f'{label}{i}.obj')
#             # shutil.copyfile(instance_path, out_file_path)

#             directions_path = osp.join(class_path, f'{label}{i}.json')
#             front = list(map(lambda x: float(x), instance['front'][0].split('\\,')))
#             up = list(map(lambda x: float(x), instance['up'][0].split('\\,')))
#             with open(directions_path, 'w') as f:
#                 json.dump({
#                     'front': front,
#                     'up': up
#                 }, f)

for label in tqdm(labels):
    if label=='fridge':
        file_path = osp.join(jsons_dir, f'{label}.json')
        with open(file_path, 'r') as f:
            json_file = json.load(f)
            instances = json_file['response']['docs']
        breakpoint()

            
