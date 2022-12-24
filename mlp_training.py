import sys
import argparse
import os
import torch
from tqdm import tqdm
from configs.config_utils import CONFIG, read_to_dict, mount_external_config
from net_utils.utils import load_dataloader
from adl_scripts.MLP_Regressor import MLP_Regressor
from torch.utils.tensorboard import SummaryWriter
from time import time

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Pose2Room.')
    parser.add_argument('--config', type=str, default='configs/config_files/p2rnet_train.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo', help='Please specify the demo path.')
    return parser.parse_args()

def validate_epoch(data_loader_validation, optimizer, model, loss_func):
    device = torch.device("cuda")
    current_loss = 0.0

    for scene_id, scene_data in enumerate(tqdm(data_loader_validation)):
        inputs, targets = scene_data['adl_input'], scene_data['adl_output']
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        predictions = model(inputs)

        loss = loss_func(predictions, targets)
        # breakpoint()
        loss.backward()

        optimizer.step()    

        
        current_loss += loss.item()
        # break
    print(f'Validation Loss: {current_loss}')
    return current_loss


def main():
    writer = SummaryWriter()
    
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
    dataset = 'MLP'
    cfg = mount_external_config(cfg)
    train_loader = load_dataloader(cfg, mode='train', dataset=dataset)
    data_loader = train_loader.dataloader
    data_loader_validation = load_dataloader(cfg, mode='val', dataset=dataset).dataloader

    device = torch.device("cuda")
    epochs = 1000
    #input_size = 768*53*3 + 10*8
    input_size = 10*8
    output_size = 10*1024
    layer_sizes = [1024]
    model = MLP_Regressor(input_size=input_size, output_size=output_size, layer_sizes=layer_sizes)
    
    model = model.to(device)
    model.train()
    
    l2_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    val_epoch = 3

    # train
    
    for epoch in range(epochs):
        start = time()
        current_loss = 0.0

        for scene_id, scene_data in enumerate(tqdm(data_loader)):
        # for scene_data in [temp_data]:
            inputs, targets = scene_data['adl_input'], scene_data['adl_output']
            breakpoint()
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            predictions = model(inputs)
            
            loss = l2_loss(predictions, targets)
            # breakpoint()
            loss.backward()

            optimizer.step()    

            
            current_loss += loss.item()
            # break
        print(f'Loss: {current_loss}')

        writer.add_scalar("current_loss", current_loss, epoch)
        writer.add_scalar("Loss/train", loss, epoch)
        end = time()
        print(f'epoch time: {end-start}')
        
        if epoch % val_epoch == 0:
            # validation
            print("Validation...\tEpoch:", epoch)
            validation_loss = validate_epoch(data_loader_validation, optimizer, model, l2_loss)
            writer.add_scalar("Loss/validation", validation_loss, epoch)
        
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
