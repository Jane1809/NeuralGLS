# NeuralGLS: Learning to Guide Local Search with Graph Convolutional Network for the Traveling Salesman Problem

Code accompanying the paper [NeuralGLS: Learning to Guide Local Search with Graph Convolutional Network for the Traveling Salesman Problem](https://link.springer.com/article/10.1007/s00521-023-09042-6).


## Dependencies
python 3.8.13
Torch
Numpy
tqdm
pyconcorde
pylkh


## Datasets
generate solved TSP instances:
```
python generate_instances.py <number of instances to generate> <number of nodes> <dataset directory>
```

Then, prepare the dataset using:
```
python preprocess_dataset.py <dataset directory>
```

## Training
Train the model using:
```
python train.py <dataset directory> <tensorboard directory> --use_gpu --config <configs/tsp<number of nodes>.json>
```

## Testing
Evaluate the model using:
```
python test.py <dataset directory>/test.txt <checkpoint path> <run directory> regret_pred --use_gpu --config <configs/tsp<number of nodes>.json>
```

## Citation
If you this code is useful in your research, please cite our paper:
```
@article{sui2023neuralgls,
  title={NeuralGLS: learning to guide local search with graph convolutional network for the traveling salesman problem},
  author={Sui, Jingyan and Ding, Shizhe and Xia, Boyang and Liu, Ruizhi and Bu, Dongbo},
  journal={Neural Computing and Applications},
  pages={1--20},
  year={2023},
  publisher={Springer}
}
```

This project expands upon the groundwork established in the previous study "Graph Neural Network Guided Local Search for the Traveling Salesperson Problem."
