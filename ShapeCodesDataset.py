import sys
import argparse
import os
from configs.config_utils import CONFIG, read_to_dict, mount_external_config
from net_utils.utils import load_dataloader

import sys
sys.path.insert(0,'/home/gogebakan/workspace/Pose2Room')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Pose2Room.')
    parser.add_argument('--config', type=str, default='configs/config_files/p2rnet_train.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo', help='Please specify the demo path.')
    return parser.parse_args()

def main():
    args = parse_args()
    config = read_to_dict('/home/gogebakan/workspace/Pose2Room/configs/config_files/p2rnet_train.yaml')
    # initiate device environments
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']['gpu_ids']
    from net_utils.utils import initiate_environment, get_sha
    config = initiate_environment(config)

    # initialize config
    cfg = CONFIG(args, config)
    cfg.update_config(args.__dict__)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string("git:\n  {}\n".format(get_sha()))
    cfg.log_string(cfg.config)
    cfg.write_config()
    
    '''Mount external config data'''
    cfg = mount_external_config(cfg)
    train_loader = load_dataloader(cfg, mode='train')
    data_loader = train_loader.dataloader

    for iter, data in enumerate(data_loader):
        print(data)
        break
    
if __name__ == '__main__':
    main()
