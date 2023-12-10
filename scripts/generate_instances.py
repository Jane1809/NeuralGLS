#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np
import sys
sys.path.append('/home/suijingyan/ourNN_GLS/NeuralGLS')
import gnngls
from gnngls import datasets
import pdb


def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
    return G


def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        # pdb.set_trace()
        G = nx.Graph() 

        coords = np.random.random((n_nodes, 2)) 

        for n, p in enumerate(coords):
            G.add_node(n, pos=p)
        # pdb.set_trace()

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos']) 
            G.add_edge(i, j, weight=w) 
        # pdb.set_trace()

        opt_solution = gnngls.optimal_tour(G, scale=1e6)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

 

def get_solved_instances_sjy(n_nodes, n_instances):
    # pdb.set_trace()
    G = nx.Graph() 
    coords = np.random.random((n_nodes, 2)) 
    for n, p in enumerate(coords):
        G.add_node(n, pos=p)
    # pdb.set_trace()
    for i, j in itertools.combinations(G.nodes, 2):
        w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos']) 
        G.add_edge(i, j, weight=w)
    # pdb.set_trace()
    opt_solution = gnngls.optimal_tour(G, scale=1e6)
    in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
    nx.set_edge_attributes(G, in_solution, 'in_solution')
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    if args.dir.exists():
        raise Exception(f'Output directory {args.dir} exists.')
    else:
        args.dir.mkdir()
        
    
    # pool = mp.Pool(processes=None)
    pool = mp.Pool(processes=60)
    instance_gen = get_solved_instances(args.n_nodes, args.n_samples)
    # pdb.set_trace()
    for G in pool.imap_unordered(prepare_instance, instance_gen): 
        nx.write_gpickle(G, args.dir / f'{uuid.uuid4().hex}.pkl')
    pool.close()
    pool.join()



