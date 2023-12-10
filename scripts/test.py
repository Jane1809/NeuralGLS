#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import pathlib
import time
import uuid

import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

import sys
sys.path.append('/home/suijingyan/ourNN_GLS/NeuralGLS')

import gnngls

# sjy 
# from gnngls import models
from gnngls import algorithms, datasets
import time
import pdb
from gnngls.gcn_model import ResidualGatedGCNModel
from torch.autograd import Variable
import torch.nn.functional as F
from config import *
from utils.graph_utils import *
from utils.model_utils import *
import os
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('model_path', type=pathlib.Path)
    parser.add_argument('run_dir', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('--time_limit', type=float, default=10.)
    parser.add_argument('--perturbation_moves', type=int, default=20)
    parser.add_argument('--use_gpu', action='store_true')
    # sjy 
    parser.add_argument('-c','--config', type=str, default="configs/default.json")

    args = parser.parse_args()

    params = json.load(open(args.model_path.parent / 'params.json'))
    # sjy    
    config_path = args.config
    config = get_config(config_path)
    print("Loaded {}:\n{}".format(config_path, config))
    print("params: \n", params)

    # pdb.set_trace()
    # sjy 
    num_nodes = config.num_nodes
    num_neighbors = int((20/math.log(20))*math.log(num_nodes))
    num_neighbors = min(num_neighbors, num_nodes-1)
    num_neighbors = int(0.8 * num_neighbors) 

    # pdb.set_trace()

    # sjy    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)  
    # sjy 
    test_set = datasets.TSPDataset(args.data_path)

    # pdb.set_trace()
    if 'regret_pred' in args.guides:
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


        model = ResidualGatedGCNModel(config, dtypeFloat, dtypeLong).to(device)

        checkpoint = torch.load(args.model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    pbar = tqdm.tqdm(test_set.instances)
    gaps = []
    search_progress = []
    time_encoder = []
    time_gls = []

    for instance in pbar:
        G = nx.read_gpickle(test_set.root_dir / instance)

        opt_cost = gnngls.optimal_cost(G, weight='weight')

        time_1 = time.time()
        t = time.time()
        search_progress.append({
            'instance': instance,
            'time': t,
            'opt_cost': opt_cost
        })

        if 'regret_pred' in args.guides:
         
            H = datasets.get_scaled_features(G, num_neighbors, test_set.scalers)
            # pdb.set_trace()
            x_nodes_coord, x_edges_values, x_edges, y_edges_regret = H[0].unsqueeze(0).to(device), H[1].unsqueeze(0).to(device), H[2].unsqueeze(0).to(device), H[3].unsqueeze(0).to(device)

            with torch.no_grad():
                # y_pred = model(H, x)
                y_pred = model(x_nodes_coord, x_edges_values, x_edges)

            num_nodes = y_pred.size(1)
            y_pred_regretmatrix = y_pred.reshape(num_nodes,-1)

            # pdb.set_trace()
            edge_list = list(G.edges)
            for i in range(len(G.edges)):
                regret_pred_value = (y_pred_regretmatrix[edge_list[i][0]][edge_list[i][1]]+y_pred_regretmatrix[edge_list[i][1]][edge_list[i][0]])/2 #预测的regret矩阵上三角与下三角对称位置的值取加和平均
                G.edges[edge_list[i]]['regret_pred'] = np.maximum(regret_pred_value.cpu().numpy(), 0)
            
            # sjy # When using a small model to test larger-scale randomly generated problems or larger-scale tsplib problems (scale larger than 190), use 'weight' to generate an initial tour instead.
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')  
            # init_tour = algorithms.nearest_neighbor(G, 0, weight='weight')


        else:
            init_tour = algorithms.nearest_neighbor(G, 0, weight='weight')

        init_cost = gnngls.tour_cost(G, init_tour)
        time_2 = time.time()
        time_encoder.append(time_2-time_1)
        best_tour, best_cost, search_progress_i = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                 t + args.time_limit, weight='weight',
                                                                                 guides=args.guides,
                                                                                 perturbation_moves=args.perturbation_moves,
                                                                                 first_improvement=False)
        time_3 = time.time()
        time_gls.append(time_3-time_2)
        for row in search_progress_i:
            row.update({
                'instance': instance,
                'opt_cost': opt_cost
            })
            search_progress.append(row)

        gap = (best_cost / opt_cost - 1) * 100
        # pdb.set_trace()
        # print(gap)

        gaps.append(gap)
        pbar.set_postfix({
            'Avg Gap': '{:.4f}'.format(np.mean(gaps)),
        })
        pbar.update(1)

    pbar.close()

    search_progress_df = pd.DataFrame.from_records(search_progress)
    search_progress_df['best_cost'] = search_progress_df.groupby('instance')['cost'].cummin()
    search_progress_df['gap'] = (search_progress_df['best_cost'] / search_progress_df['opt_cost'] - 1) * 100
    search_progress_df['dt'] = search_progress_df['time'] - search_progress_df.groupby('instance')['time'].transform(
        'min')
    # pdb.set_trace()
    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # sjy 
    # run_name = f'{timestamp}_{uuid.uuid4().hex}.pkl'
    batch_size = params['batch_size']
    # sjy 
    time_limit = int(args.time_limit)
    model_size = params['model_size']
    run_name = f'{"P"}{num_nodes}_{"M"}{model_size}_{"B"}{batch_size}_{"TL"}{time_limit}_{timestamp}_{uuid.uuid4().hex}.pkl'
    if not args.run_dir.exists():
        args.run_dir.mkdir()
    if not (args.run_dir/config.expt_name).exists():
        (args.run_dir/config.expt_name).mkdir()
    # search_progress_df.to_pickle(args.run_dir / run_name)
    search_progress_df.to_pickle(args.run_dir / config.expt_name / run_name)

    # print(time_encoder)
    # print(time_gls)
    t1 = sum(time_encoder)
    t2 = sum(time_gls)
    print(t1)
    print(t2)
    print("Time percent for network: ", t1/(t1+t2))
    print("Time percent for gls: ", t2/(t1+t2))
    print(config.expt_name)
    print(run_name)
    print("time_limit=", args.time_limit)
    print(args.data_path)
    print(args.model_path)

