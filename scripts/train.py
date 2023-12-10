#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
from functools import partial
import json
import os
import pathlib
import uuid

import dgl.nn
import torch
import tqdm.auto as tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/home/suijingyan/ourNN_GLS/NeuralGLS')
from gnngls import datasets
import pdb

# sjy 2022.10.3
# from gnngls import models
from gnngls.gcn_model import ResidualGatedGCNModel
from torch.autograd import Variable
import torch.nn.functional as F
from config import *
from utils.graph_utils import *
from utils.model_utils import *
import math




# sjy 
def train(model, data_loader, target, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(data_loader):

        # sjy 
        x_nodes_coord, x_edges_values, x_edges, y_edges_regret =  batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

        optimizer.zero_grad()

        # sjy 
        y_pred = model(x_nodes_coord, x_edges_values, x_edges)
        # sjy 
        # sjy compute loss
        batch_size = y_pred.size(0)
        num_nodes = y_pred.size(1)
        y_pred_regretmatrix = y_pred.reshape(batch_size, num_nodes,-1) # B x V x V
        
        mask_diag = torch.eye(num_nodes).to(device) #对角阵

        y_pred_regret = y_pred_regretmatrix - y_pred_regretmatrix*mask_diag #去掉对角线上的元素

        loss = criterion(y_pred_regret, y_edges_regret)
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    # pdb.set_trace()
    epoch_loss /= (batch_i + 1)
    return epoch_loss


def test(model, data_loader, target, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            x_nodes_coord, x_edges_values, x_edges, y_edges_regret =  batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            # y_pred = model(batch, x)
            y_pred = model(x_nodes_coord, x_edges_values, x_edges)

            batch_size = y_pred.size(0)
            num_nodes = y_pred.size(1)
            y_pred_regretmatrix = y_pred.reshape(batch_size, num_nodes,-1) # B x V x V
            
            mask_diag = torch.eye(num_nodes).to(device) #对角阵

            y_pred_regret = y_pred_regretmatrix - y_pred_regretmatrix*mask_diag #去掉对角线上的元素

            loss = criterion(y_pred_regret, y_edges_regret)

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        return epoch_loss


def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'val_loss': val_loss
    }, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=pathlib.Path, help='Where to load dataset')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Where to log Tensorboard data')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint frequency')
    parser.add_argument('--target', type=str, default='regret', choices=['regret', 'in_solution'])
    parser.add_argument('--use_gpu', action='store_true')
    # sjy 
    parser.add_argument('-c','--config', type=str, default="configs/default.json")

    args = parser.parse_args()

    # sjy     
    config_path = args.config
    config = get_config(config_path)
    print("Loaded {}:\n{}".format(config_path, config))
    # sjy 
    print("params: \n", dict(vars(args)))


    # sjy 
    num_nodes = config.num_nodes
    num_neighbors = int((20/math.log(20))*math.log(num_nodes))

    num_neighbors = min(num_neighbors, num_nodes-1)

   # sjy 2022.10.6     
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)  

    # Load dataset
    # pdb.set_trace()
    # # sjy  
    train_set = datasets.TSPDataset(args.data_dir / 'train.txt')
    val_set = datasets.TSPDataset(args.data_dir / 'val.txt')

    # use GPU if it is available
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    # print('device =', device)
    # sjy 

    if torch.cuda.is_available():
        print("CUDA available, using GPU ID {}".format(config.gpu_id))
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)

    else:
        print("CUDA not available")
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)

    # pdb.set_trace()



    # sjy   
    model = ResidualGatedGCNModel(config, dtypeFloat, dtypeLong).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.target == 'regret':
        criterion = torch.nn.MSELoss()

    elif args.target == 'in_solution': 
        # only works for a homogenous dataset
        y = train_set[0].ndata['in_solution']
        pos_weight = len(y) / y.sum() - 1
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # sjy 
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=partial(
            datasets.collate_fn_for_knn,
            num_neighbors=num_neighbors,
            scalers=train_set.scalers,
            train=True
        )
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=partial(
            datasets.collate_fn_for_knn,
            num_neighbors=num_neighbors,
            scalers=train_set.scalers,
            train=False
        )
    )

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{"B"}{args.batch_size}_{timestamp}_{uuid.uuid4().hex}'
    # sjy 2022.10.6
    log_dir = args.tb_dir / config.expt_name/ run_name
    writer = SummaryWriter(log_dir)

    # early stopping
    best_score = None
    counter = 0

    pbar = tqdm.trange(args.n_epochs)
    for epoch in pbar:
        
        # pdb.set_trace()
        epoch_loss = train(model, train_loader, args.target, criterion, optimizer, device)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_val_loss = test(model, val_loader, args.target, criterion, device)
        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(epoch_val_loss),
        })

        if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_{epoch}.pt'
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / checkpoint_name)

        if best_score is None or epoch_val_loss < best_score - args.min_delta:
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_best_val.pt')

            best_score = epoch_val_loss
            counter = 0
        else:
            counter += 1

        if counter >= args.patience:
            pbar.close()
            break

        lr_scheduler.step()

    writer.close()

    params = dict(vars(args))
    # sjy 
    params['model_size'] = num_nodes

    params['data_dir'] = str(params['data_dir'])
    params['tb_dir'] = str(params['tb_dir'])

    # sjy 
    json.dump(params, open(log_dir / 'params.json', 'w'))

    save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_final.pt')
