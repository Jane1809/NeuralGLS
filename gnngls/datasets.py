import copy
import pathlib
import pickle

import dgl
import networkx as nx
import numpy as np
import torch
import torch.utils.data

from . import tour_cost, fixed_edge_tour, optimal_cost as get_optimal_cost
# sjy 
import pdb
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform
import random




def set_features(G):
    for e in G.edges:
        i, j = e

        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)


def set_labels(G):
    # pdb.set_trace()
    optimal_cost = get_optimal_cost(G)

    for e in G.edges:
        regret = 0.

        if not G.edges[e]['in_solution']:
            tour = fixed_edge_tour(G, e, scale=1e6, max_trials=100, runs=10)
            cost = tour_cost(G, tour)
            regret = (cost - optimal_cost) / optimal_cost

        G.edges[e]['regret'] = regret


class TSPDataset(torch.utils.data.Dataset):
    # sjy 
    def __init__(self, instances_file, scalers_file=None):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent
        # pdb.set_trace()
        self.instances = [line.strip() for line in open(instances_file)]


        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        scalers = pickle.load(open(scalers_file, 'rb'))
        # print(scalers)
        if 'edges' in scalers: # for backward compatability
            self.scalers = scalers['edges']
        else:
            self.scalers = scalers

        # only works for homogenous datasets
        G = nx.read_gpickle(self.root_dir / self.instances[0])
        # sjy 
        self.num_nodes = len(G.nodes)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        # pdb.set_trace()

        return G



def get_scaled_features(G, num_neighbors, scalers):

    #sjy 
    # pdb.set_trace()
    
    x_edges = []           
    x_edges_values = []    
    x_nodes_coord = []     
    y_edges_regret = []    
    x_nodes_coord = []     

    for idx in range(G.number_of_nodes()):
        x_nodes_coord.append(G.nodes[idx]['pos'].tolist()) 

    # Compute distance matrix
    x_edges_values = squareform(pdist(x_nodes_coord, metric='euclidean')) 

    # Compute adjacency matrix
    if num_neighbors == -1:
        x_edges = np.ones((G.number_of_nodes(), G.number_of_nodes()))  # Graph is fully connected
    else:
        x_edges = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
        knns = np.argpartition(x_edges_values, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]
        # Make connections 
        for idx in range(G.number_of_nodes()):
            x_edges[idx][knns[idx]] = 1
    np.fill_diagonal(x_edges, 2)  # Special token for self-connections

    edge_list = list(G.edges)
    # pdb.set_trace()

    y_edges_regret = torch.zeros(G.number_of_nodes(), G.number_of_nodes())

    # sjy 
    regret = []
    for i in range(len(G.edges)):
        regret.append(G.edges[edge_list[i]]['regret'])

    regret_transformed = scalers['regret'].fit_transform(np.expand_dims(regret, axis=1))
    
    # print(regret_transformed)


    for i in range(len(G.edges)):
        y_edges_regret[edge_list[i][0]][edge_list[i][1]] = regret_transformed[i][0]
        y_edges_regret[edge_list[i][1]][edge_list[i][0]] = regret_transformed[i][0]

    return torch.Tensor(x_nodes_coord), torch.Tensor(x_edges_values), torch.LongTensor(x_edges), y_edges_regret



def collate_fn_for_knn(Gs, num_neighbors, scalers, train=False):

    batch_x_nodes_coord = []
    batch_x_edges_values = []
    batch_x_edges = []
    batch_y_edges_regret = []

    if train:
        # pdb.set_trace()
        num_neighbors = random.randint(int(0.5*num_neighbors),num_neighbors)
    else:
        # sjy 
        num_neighbors = int(0.8 * num_neighbors)

    for G in Gs:
        x_nodes_coord, x_edges_values, x_edges, y_edges_regret = get_scaled_features(G, num_neighbors, scalers)
        batch_x_nodes_coord.append(x_nodes_coord)
        batch_x_edges_values.append(x_edges_values)
        batch_x_edges.append(x_edges)
        batch_y_edges_regret.append(y_edges_regret)

    batch_x_nodes_coord = torch.stack(batch_x_nodes_coord)
    batch_x_edges_values = torch.stack(batch_x_edges_values)
    batch_x_edges = torch.stack(batch_x_edges)
    batch_y_edges_regret = torch.stack(batch_y_edges_regret)

    return batch_x_nodes_coord, batch_x_edges_values, batch_x_edges, batch_y_edges_regret