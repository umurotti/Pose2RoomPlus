import wget
import os
import os.path as osp
import urllib.request
from pathlib import Path
from tqdm import tqdm

out_dir = '/home/baykara/adl4cv/pointnet_pytorch/data/solr_jsons'
urls = {
    'bench': 'http://localhost:8983/solr/ShapeNet/select?fl=synsetId%2C%20fullId%2C%20wnlemmas&fq=-wnlemmas%3A%20%22window%20seat%22&fq=-wnlemmas%3A%20laboratory&fq=-wnlemmas%3A%22park%20bench%22&fq=-wnlemmas%3A%22pew%22&indent=true&q.op=OR&q=wnlemmas%3A%20bench&rows=10000&useParams=&wnlemmas=bench',
    'cabinet': None, 
    'faucet': None,
    'stove': None,
    'bookshelf': None,
    'computer': None,
    'desk': None,
    'chair': None,
    'monitor': None,
    'sofa': None,
    'lamp': None,
    'nightstand': None, 
    'bed': None,
    'dishwasher': None, 
    'fridge': None,
    'microwave': None,
    'toilet': None
}

Path(out_dir).mkdir(parents=True, exist_ok=True)

for label,url in tqdm(urls.items()):
    if url:
        urllib.request.urlretrieve(url, osp.join(out_dir, f'{label}.json'))

