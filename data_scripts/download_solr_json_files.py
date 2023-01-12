import os
import os.path as osp
import urllib.request
from pathlib import Path
from tqdm import tqdm

out_dir = '/home/baykara/adl4cv/pointnet_pytorch/data/solr_jsons'
urls = {
    'bench': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&fq=-wnlemmas%3A%20%22window%20seat%22&fq=-wnlemmas%3A%20laboratory&fq=-wnlemmas%3A%22park%20bench%22&fq=-wnlemmas%3A%22pew%22&indent=true&q.op=OR&q=wnlemmas%3A%20bench&rows=10000&useParams=&wnlemmas=bench',
    'cabinet': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&fq=-name%3Arefrigerator&indent=true&q.op=OR&q=wnlemmas%3Adresser&rows=10000&useParams=',
    'faucet': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20faucet&rows=10000&useParams=&wnlemmas=bench&wt=json',
    'stove': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20stove&rows=10000&useParams=&wnlemmas=bench&wt=json',
    'bookshelf': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20bookshelf&rows=10000&useParams=&wnlemmas=bench&wt=json',
    'computer': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20laptop&rows=10000&useParams=&wnlemmas=bench&wt=json',
    'desk': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&fq=-wnlemmas%3Acabinet&indent=true&q.op=OR&q=wnlemmas%3A%20desk&rows=10000&useParams=&wnlemmas=bench&wt=json',
    'chair': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20%22straight%20chair%22&rows=10000&useParams=&wnlemmas=bench&wt=json',
    'monitor': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=name%3Atelevision%20AND%20wnlemmas%3ALCD&rows=10000&useParams=',
    'sofa': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20sofa&rows=10000&useParams=&wt=json',
    'lamp': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3A%20%22table%20lamp%22&rows=10000&useParams=&wt=json',
    'nightstand': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=name%3Anightstand%20OR%20wnlemmas%3A%22tall%20cabinet%22%20OR%20wnlemmas%3A%22desk%20cabinet%22&rows=10000&useParams=',
    'bed': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&fq=-wnlemmas%3Acabinet&indent=true&q.op=OR&q=wnlemmas%3A%20bed&rows=10000&useParams=&wt=json',
    'dishwasher': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3Adishwasher&rows=10000&useParams=',
    'fridge': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=name%3Arefrigerator&rows=10000&useParams=',
    'microwave': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=wnlemmas%3Amicrowave&rows=10000&useParams=',
    'toilet': 'http://localhost:8983/solr/ShapeNet/select?fl=front%2C%20up%2C%20synsetId%2C%20fullId%2C%20wnlemmas&indent=true&q.op=OR&q=name%3Atoilet&rows=10000&useParams='
}

Path(out_dir).mkdir(parents=True, exist_ok=True)

for label,url in tqdm(urls.items()):
    if url:
        urllib.request.urlretrieve(url, osp.join(out_dir, f'{label}.json'))

