import sys
import argparse
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
from time import time
import shutil

from torch.utils.tensorboard import SummaryWriter

from configs.config_utils import CONFIG, read_to_dict, mount_external_config
from net_utils.utils import load_dataloader
from adl_scripts.MLP_Regressor import MLP_Regressor



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Pose2Room.')
    parser.add_argument('--config', type=str, default='configs/config_files/p2rnet_train.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo', help='Please specify the demo path.')
    return parser.parse_args()


def train_epoch(dataloader, optimizer, model, loss_func):
    model.train()
    device = torch.device("cuda")
    current_loss = 0.0

    for batch_data in dataloader:
        inputs, targets = batch_data['adl_input'], batch_data['adl_output']
        inputs = Variable(inputs.to(device))
        targets = Variable(targets.to(device))
        
        optimizer.zero_grad()
        
        predictions = model(inputs)
        
        loss = loss_func(predictions, targets)
        
        loss.backward()

        optimizer.step()    
        
        current_loss += loss.item()
    
    return current_loss

def validate_epoch(dataloader, model, loss_func):
    model.eval()
    device = torch.device("cuda")
    current_loss = 0.0

    with torch.no_grad():
        for batch_data in dataloader:
            inputs, targets = batch_data['adl_input'], batch_data['adl_output']
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            loss = loss_func(predictions, targets)
            
            current_loss += loss.item()
            
    return current_loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    train_loader = load_dataloader(cfg, mode='train', dataset=dataset).dataloader
    validation_loader = load_dataloader(cfg, mode='val', dataset=dataset).dataloader

    device = torch.device("cuda")
    epochs = 200
    val_epoch = 3
    nearest_k_frames = 10
    input_size = 8 + 2*nearest_k_frames*256
    output_size = 1024
    layer_sizes = [2048, 1024]
    model = MLP_Regressor(input_size=input_size, output_size=output_size, layer_sizes=layer_sizes)
    
    model = model.to(device)
    model.train()
    
    l2_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = 10*5
    patience = 10
    early_stop_counter = 0

    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        #train
        start = time()
        train_loss = train_epoch(train_loader, optimizer, model, l2_loss)
        end = time()
        epoch_time = end-start
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        
        # validation
        if epoch % val_epoch == 0:
            validation_loss = validate_epoch(validation_loader, model, l2_loss)
            pbar.write(f'Training loss:\t{train_loss:.5f}\t{epoch_time:.2f}s\tValidation loss:\t{validation_loss:.5f}\t{epoch_time:.2f}s')
            pbar.update(1)
            writer.add_scalar("Loss/validation_epoch", validation_loss, epoch)

            if validation_loss > best_loss:
                early_stop_counter += 1
                if early_stop_counter == patience:
                    print(f'early stopping with best validation_loss: {validation_loss}')
                    break
            else:
                best_loss = validation_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f'saved_models/checkpoint.pt')
                    

        else:
            pbar.write(f'Training loss:\t{train_loss:.5f}\t{epoch_time:.2f}s')
            pbar.update(1)
        

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
