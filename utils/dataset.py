from typing import Callable, List, Optional

import torch
import math
import pickle
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import CitationFull
# from torch_geometric.datasets import Actor

from torch_geometric.nn import APPNP
from torch_sparse import coalesce

from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):
        if name=='actor':
            name='film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.train_percent = self.data.train_percent.item()
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


r'''Note:
    Override `PYG` Actor dataset'''
class Actor(InMemoryDataset):

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, 
                 root: str,
                 name: str = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        if not osp.isdir(root):
            os.makedirs(root)
        self.root = root
        self.name = name
        super(Actor, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    # @property
    # def raw_file_names(self):
    #     return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
        # for f in self.raw_file_names[:2]:
        #     download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        # for f in self.raw_file_names[2:]:
        #     download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            row, col = torch.tensor(rows), torch.tensor(cols)

            x = torch.zeros(int(row.max()) + 1, int(col.max()) + 1)
            x[row, col] = 1.

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            from torch_geometric.utils import coalesce    
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f, allow_pickle=True)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)




def DataLoader(name):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = dataset_heterophily(root='./data/', name=name, transform=T.NormalizeFeatures())

    #* ADD CS
    elif name in ["dblp"]:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = CitationFull(path, name, transform=T.NormalizeFeatures())
    elif name in ["cs", "physics"]:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Coauthor(path, name, transform=T.NormalizeFeatures())
        
    elif name in ['texas', 'cornell']:
        dataset = WebKB(root='./data/',name=name, transform=T.NormalizeFeatures())

    # TODO
    elif name in ['actor']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Actor(root='./data/', name=name, transform=T.NormalizeFeatures())
    
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset

"""Use PYG node random split function to split dataset"""
def std_random_dataset(name, 
                       num_train_per_class=20, 
                       num_val=500, 
                       num_test=1000):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.RandomNodeSplit(split="random", 
                                                                    num_train_per_class=num_train_per_class, 
                                                                    num_val=num_val, 
                                                                    num_test=num_test))
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, transform=T.RandomNodeSplit(split="random", 
                                                       num_train_per_class=num_train_per_class, 
                                                       num_val=num_val, 
                                                       num_test=num_test))
    elif name in ['chameleon', 'squirrel']:
        dataset = dataset_heterophily(root='./data/', name=name, transform=T.RandomNodeSplit(split="random", 
                                                                                            num_train_per_class=num_train_per_class, 
                                                                                            num_val=num_val, 
                                                                                            num_test=num_test))
    # TODO
    #* ADD CS
    elif name in ["cs", 'physics']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Coauthor(path, name, transform=T.RandomNodeSplit(split="random", 
                                                                    num_train_per_class=num_train_per_class, 
                                                                    num_val=num_val, 
                                                                    num_test=num_test))
    # TODO
    elif name in ["actor"]:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Actor(root="./data/", name=name, transform=T.RandomNodeSplit(split="random", 
                                                                    num_train_per_class=num_train_per_class, 
                                                                    num_val=num_val, 
                                                                    num_test=num_test))        
    
    elif name in ["dblp"]:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = CitationFull(path, name, transform=T.RandomNodeSplit(split="random", 
                                                                        num_train_per_class=num_train_per_class, 
                                                                        num_val=num_val, 
                                                                        num_test=num_test))
    
    elif name in ['texas', 'cornell']:
        dataset = WebKB(root='./data/',name=name, transform=T.RandomNodeSplit(split="random", 
                                                                            num_train_per_class=num_train_per_class, 
                                                                            num_val=num_val, 
                                                                            num_test=num_test))

    elif name in ['ogbn-arxiv']:
        # transforms = None
        from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
        dataset =  PygNodePropPredDataset(name="ogbn-arxiv", root="./data/", transform=T.ToSparseTensor())
        # dataset =  PygNodePropPredDataset(name="ogbn-arxiv", root="./data/")

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset