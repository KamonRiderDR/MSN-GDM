'''
Description: 
Author: Rui Dong
Date: 2024-04-11 13:35:45
LastEditors: Please set LastEditors
LastEditTime: 2024-06-19 16:30:20
'''
import os
import math
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

# ADD?
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.utils import contains_isolated_nodes, to_dense_adj, dense_to_sparse, \
                                    degree, to_undirected, remove_self_loops
from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))

    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    return data


def ogbn_dataloader(args, data):
    """_summary_
    Refernce:
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_train.py#L42
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py
    Args:
        args (_type_): _description_
        dataset (_type_): _description_

    Returns:
        train_loader, val_loader, test_loader
    """
    
    # split_idx = data.get_idx_split()
    # data.train_mask, dataval_mask, test_idx = split_idx["train"], \
    #                                split_idx["valid"], \
    #                                split_idx["test"]    
    # train_loader    = NeighborLoader(data[0], 
    #                                 input_nodes=train_idx,
    #                                 num_neighbors=[args.fan_out] * args.n_layers,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=args.num_workers) 
    # val_loader      = NeighborLoader(data[0], 
    #                                 input_nodes=val_idx,
    #                                 num_neighbors=[args.fan_out] * args.n_layers,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=args.num_workers) 
    # test_loader     = NeighborLoader(data[0], 
    #                                 input_nodes=test_idx,
    #                                 num_neighbors=[args.fan_out] * args.n_layers,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=args.num_workers) 

    train_loader = GraphSAINTRandomWalkSampler(data,
                                               batch_size=4096,
                                               walk_length=args.n_layers,
                                               num_steps=5,
                                               num_workers=args.num_workers)

    return train_loader

def load_dataset_param(args, json_path):
    with open(json_path, 'r') as f:
        dicts = json.load(f)
        for dict in dicts:
            if dict["dataset"] == args.dataset:
                f.close()
                return dict
    f.close()
    raise NameError("unknown dataset type: ", args.dataset)


def update_dataset_param(args, dataset, item, value):
    dicts = []
    config_path = str("{}/config/param_dataset.json".format(args.root))
    with open(config_path, mode="r", encoding="utf-8") as f:
        dicts = json.load(f)
        for dict in dicts:
            if dict["dataset"] == dataset:
                dict[item] = value
    f.close()
    
    with open(config_path, mode='w', encoding="utf-8") as f:
        json.dump(dicts, f, indent=1)
    f.close()


def plot_dist_curve(args):
    """ Plot curve if different methods. Use MSE / MAD as a metric function.

    Args:
        args (_type_): TBD
    
    Return:

    """
    npy_dir = args.root + "/save/dist_metrics"
    save_dir = args.root + "/plot/" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = "{}/dist_{}_{}_{}.jpg".format(save_dir, args.gamma, args.alpha, args.J)
    
    np_tea = np.load("{}/save/dist_metrics/tea_{}_{}_dist.npy".format(args.root, args.net, args.dataset))
    np_kd = np.load("{}/KD_{}_dist.npy".format(npy_dir, args.dataset))
    np_stu = np.load("{}/SpecMLP_{}_dist.npy".format(npy_dir, args.dataset))
    min_len = min(len(np_kd), len(np_stu))
    min_len_1 = min(min_len, len(np_tea))
    x =[i+1 for i in range(min_len)]

    plt.plot(x, np_kd[:min_len], c='blue', label="KD")
    plt.plot(x, np_stu[:min_len], c='darkorange', label="Ours")
    plt.plot(x[:min_len_1], np_tea[:min_len_1], label='Teacher')
    plt.legend(["KD", "Ours", "Teacher"])
    plt.grid(alpha=0.3)
    plt.savefig(save_file)
    plt.clf()
    plt.close()


def ms_plot_curve(args):
    npy_dir = args.root + "/save/dist_metrics"
    save_dir = args.root + "/plot/" + args.dataset
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = "{}/ms_dist_{}_{}_{}.jpg".format(save_dir, args.gamma, args.alpha, args.J)
    np_kd = np.load("{}/KD_{}_dist.npy".format(npy_dir, args.dataset))
    np_stu = np.load("{}/ms_SpecMLP_{}_dist.npy".format(npy_dir, args.dataset), allow_pickle=True).tolist()
    min_len = min(len(np_kd), len(np_stu[0]))
    min_len = min(min_len, 600)
    x = [i+1 for i in range(min_len)]
    
    names = ["KD"]
    plt.plot(x, np_kd[:min_len], c='blue', label="KD")
    for i in range(len(np_stu)):
        if len(np_stu[i]) > 0:
            plt.plot(x, np_stu[i][:min_len], label="{} scale".format(i+1))
            names.append("{} scale".format(i+1))
    plt.legend(names)
    plt.grid(alpha=0.2)
    plt.savefig(save_file)
    plt.clf()
    plt.close()


def ms_plot_curve_J(args):
    """_summary_

    Args:
        args (_type_): _description_
        j (_type_): _description_\
    """
    
    npy_dir = args.root + "/save/dist_metrics"
    save_dir = args.root + "/plot/" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np_kd = np.load("{}/ms_KD_{}_dist.npy".format(npy_dir, args.dataset))
    np_stu = np.load("{}/ms_SpecMLP_{}_dist.npy".format(npy_dir, args.dataset), allow_pickle=True).tolist()

    for i in range(len(np_stu)):
        if len(np_stu[i]) > 0:
            min_len = min(len(np_kd[i]), len(np_stu[i]))
            min_len = min(min_len, 600)
            x = [i+1 for i in range(min_len)]

            save_file = "{}/ms_dist_{}_{}_{}_{}.jpg".format(
                                        save_dir, 
                                        args.gamma, 
                                        args.alpha, 
                                        args.J, 
                                        i)
            names = ["KD", i+1]
            plt.plot(x, np_kd[i][:min_len], label="KD")
            plt.plot(x, np_stu[i][:min_len], label="{} scale".format(i+1))
            plt.legend(names)
            plt.grid(alpha=0.2)
            plt.savefig(save_file)
            plt.clf()
            plt.close()



def ms_plot_curve_ff(args, data):
    """TODO THIS is for MSM && MDM PLOT.

    Args:
        args (_type_): _description_
    """
    npy_dir = args.root + "/save/dist_metrics"
    save_dir = args.root + "/plot_new/" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np_tea = np.load("{}/tea_{}_{}_dist.npy".format(npy_dir, args.net, args.dataset))
    np_stu = np.load("{}/ms_SpecMLP_{}_dist.npy".format(npy_dir, args.dataset), allow_pickle=True).tolist()

    ablations = ["KD", "GLNN", "FF-G2M"]
    np_ablations = [None for i in range(len(ablations))]

    for i in range(len(ablations)):
        np_ablations[i] = np.load("{}/ms_{}_{}_dist.npy".format(
            npy_dir, ablations[i], args.dataset))
    
    for i in range(len(np_stu)):
        coffi = data.mean_sim[i]
        if len(np_stu[i]) > 0:
            min_len = 999999
            for j in range(len(np_ablations)):
                min_len = min(min_len, len(np_ablations[j][i]))
            # min_len = min(len(np_kd[i]), len(np_stu[i]))
            min_len = min(min_len, len(np_stu[i]))
            min_len = min(min_len, 600)
            x = [i+1 for i in range(min_len)]

            save_file = "{}/ms_dist_{}_{}_{}_{}.pdf".format(save_dir, args.gamma, args.alpha, args.J, i)
            names = ["KD", "GLNN", "FF-G2M", "{}-scale".format(i+1)]

            for j in range(len(ablations)):
                plt.plot(x, np_ablations[j][i][:min_len], label="{}".format(ablations[j]))
            plt.plot(x, np_stu[i][:min_len], label="{} scale".format(i+1))

            plt.legend(names, fontsize=18)
            plt.xlabel("Epochs", fontsize=18)
            if coffi < 0:
                plt.ylabel("MSM-Metrics", fontsize=20)
            else:
                plt.ylabel("MDM-Metrics", fontsize=20)

            plt.grid(alpha=0.2)
            plt.savefig(save_file, bbox_inches="tight")
            plt.clf()
            plt.close()



"""==================================================="""
"""============= Metricss calculation ================"""
"""==================================================="""


def node_sim_metrics(args, x, edge_index):
    """ Caculate node similarity between node pairs.

    Args:
        args (_type_): _description_
        data (_type_): _description_
        edge_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion = f.cosine_similarity
    # criterion = nn.MSELoss()
    src = edge_index[0]
    dst = edge_index[1]
    # cos_sim = - criterion(x[src], x[dst]) + 1
    sim = criterion(x[src], x[dst])
    # dist = torch.add(cos_sim, 1) 
    # sim = criterion(x[src], x[dst])
    
    return torch.mean(sim) 

def node_sim_metrics_h(args, x, x_kd, edge_index):
    """node similarity metric of different node pairs. Use MSE as metric function.

    Args:
        args:
        x:          x_student
        x_kd:       x_kd (x_teacher)
        edge_index: _description_
    """
    criterion = nn.MSELoss()
    src = edge_index[0]
    dst = edge_index[1]
    z_1 = f.softmax(abs(x[src]-x[dst]), dim=-1)
    z_2 = f.softmax(abs(x_kd[src]- x_kd[dst]), dim=-1)
    
    sim_h = criterion(z_1, z_2)
    return sim_h 


def ms_node_sim_metrics(args, x,  data):
    """_summary_

    Args:
        args (_type_): _description_
        x (_type_): _description_
        data (_type_): _description_

    Returns:
        `list` format:  J * 1
    """
    multi_edge_index = []
    for i in range(args.J):
        multi_edge_index.append(getattr(data, "{}_edge_index".format(i)))
    
    ms_node_sim = []
    for i in range(args.J):
        ms_node_sim.append(node_sim_metrics(args, x, multi_edge_index[i]))
    
    return  ms_node_sim


def ms_node_sim_metrics_ff(args, data, x, x_tea = None):
    """Return Different  node pairs. If coffi < 0, then similar metrics. Else return different metrics. 

    Args:
        args (_type_): _description_
        data (_type_): _description_
        x (_type_): _description_
        x_kd (_type_, optional): _description_. Defaults to None.

    Returns:
        `list` format:  J * 1  (sim && diff metrics)
    """
    multi_edge_index = []
    for i in range(args.J):
        multi_edge_index.append(getattr(data, "{}_edge_index".format(i)))
    
    ms_node_sim = []
    for i in range(args.J):
        coffi = data.mean_sim[i]
        if coffi < 0:
            ms_node_sim.append(node_sim_metrics(args, x, multi_edge_index[i]))
        else:
            ms_node_sim.append(node_sim_metrics_h(args, x, x_tea, multi_edge_index[i]))
            
    return  ms_node_sim



"""====================================================="""
"""========== Below are preprocess methods. ============"""
"""====================================================="""

#! TODO
def check_index(J, similarity):
    """check index in `multi_scale_node_sampler()`

    Args:
        J (_type_): _description_
        similarity (_type_): _description_

    Returns:
        k index of similarity. k==0 for similarity == 1 and k == J-1 for sim == -1
    """
    end = 1
    start = end - 2 / (J-1)
    for k in range(J):
        if similarity > start and similarity <= end:
            return k
        
        end = start
        start -= 2 / (J-1)

    return J-1

def check_index_homo(J, similarity):
    """return index of similarity. Index = 0 if similarity = -1 and Index = J if sim = 1"""
    end = -1
    start = end + 2 / (J)
    for k in range(J):
        if similarity < start and similarity >= end:
            return k
        
        end = start
        start += 2 / (J)

    return J-1


# TODO
def multi_scale_node_sampler(args, datalist):
    """multi scale node samplers. (J-scale according to cos similarities)
        Similarity among [-1, 1]. -1 for most similar, 1 for different

    Args:
        args (_type_): _description_
        datalist (_type_): _description_

    Returns:
        datalist after preprocessing
    """
    
    J = args.J
    data_list_ = []
    
    for data in datalist:
        if "ogbn" not in args.dataset:
            '''preprocess isolated nodes'''
            if contains_isolated_nodes(data.edge_index, num_nodes=data.x.size(0)):
                adj = to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.x.size(0))[0]
                deg = torch.sum(adj, dim=1)
                no_deg = (deg == 0)
                no_deg_index = no_deg.nonzero(as_tuple=True)[0]
                ones = torch.eye(adj.size(0), device=data.edge_index.device)
                adj[no_deg_index] = ones[no_deg_index]
                deg = torch.sum(adj, dim=1)
                
                data.edge_index, _ = dense_to_sparse(adj)
        else:
            data.adj_t = data.adj_t.to_symmetric()
            row, col,_ = data.adj_t.t().coo()
            data.edge_index = torch.stack([row, col], axis=0)

        x, edge_index, y = data.x, data.edge_index, data.y
        edge_index = to_undirected(edge_index)  
        deg = degree(edge_index[0])
        # deg += 1 # if zero
        '''calculate ego-node homo-ratio'''
        homo_num = np.zeros(deg.shape)
        for edge in edge_index.t():
            sim_edge = f.cosine_similarity(x[edge[0]], x[edge[1]], dim=0)
            homo_num[edge[0]] += sim_edge
            homo_num[edge[1]] += sim_edge
        homo_num = torch.tensor(homo_num)
        deg = torch.divide(homo_num, deg)   # [-1, 1]
        
        multi_edge_index = []
        for i in range(args.J):
            multi_edge_index.append(torch.empty(0, 2, dtype=edge_index.dtype))
        mean_sim = [0 for i in range(args.J)]
        total_sim = [ []  for i in range(args.J)]


        for edge in edge_index.t():
            #! TODO
            '''cos-sim + node-homo-sim'''
            sim = f.cosine_similarity(x[edge[0]], x[edge[1]], dim=0)
            flag = sim / 2.0 + 0.5
            src_val = deg[edge[0]]
            dst_val = deg[edge[1]]

            c_1 = - sim                                         # [-1, 1]
            c_2 = abs(dst_val - src_val)                        # [0, 2]
            alpha = args.alpha
            sim_edge = alpha*c_1 + (1-alpha)*c_2 + (alpha-1)    # [-1, 1]
            idx = check_index_homo(J, sim_edge)

            if idx < J:
                multi_edge_index[idx] = torch.cat((multi_edge_index[idx], edge.view(1, -1)), dim=0)
                total_sim[idx].append(sim_edge.detach().cpu())
            else:
                multi_edge_index[0] = torch.cat((multi_edge_index[0], edge.view(1, -1)), dim=0)
                total_sim[0].append(sim_edge.detach().cpu())

        multi_edge_index = [edge_index.t() for edge_index in multi_edge_index]
        for i in range(J):
            setattr(data, "{}_edge_index".format(i), multi_edge_index[i])
            if len(total_sim[i]) > 0:
                mean_sim[i] = sum(total_sim[i]) / len(total_sim[i])

        # print(mean_sim)
        data.mean_sim = mean_sim

        data_list_.append(data)

    return data_list_


@torch.no_grad()
def multi_scale_edge_sampler(args, data):
    print("Begin multi_scale_edge_sampler")
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    J = args.J

    print(args.dataset)
    print(data)
    # srcs, dsts = data.all_edges()
    # data.add_edges(dsts, srcs)
    # data = data.remove_self_loop().add_self_loop()


    # if "ogbn" in args.dataset:
    #     data.adj_t = data.adj_t.to_symmetric()
    #     row, col,_ = data.adj_t.t().coo()
    #     data.edge_index = torch.stack([row, col], axis=0)
    
    # data.edge_index = remove_self_loops(data.edge_index)
    '''preprocess isolated nodes'''
    if contains_isolated_nodes(data.edge_index, num_nodes=data.x.size(0)):
        adj = to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.x.size(0))[0]
        deg = torch.sum(adj, dim=1)
        no_deg = (deg == 0)
        no_deg_index = no_deg.nonzero(as_tuple=True)[0]
        ones = torch.eye(adj.size(0), device=data.edge_index.device)
        adj[no_deg_index] = ones[no_deg_index]
        deg = torch.sum(adj, dim=1)
        data.edge_index, _ = dense_to_sparse(adj)    
    x, edge_index, y = data.x, data.edge_index, data.y
    
    # print(edge_index[0].shape)
    # num_edge_sampler = int(edge_index.shape[1] * 0.2)
    num_edge_sampler = int( data.num_nodes * 3.5 )
    print("num_edge_sampler: {}".format(num_edge_sampler))
    # num_edge_sampler = 238162
    edge_index = random_sample_edges(edge_index, num_edge_sampler)
    edge_index = to_undirected(edge_index)  
    deg = degree(edge_index[0])
    # deg += 1 # if zero
    '''calculate ego-node homo-ratio'''
    homo_num = np.zeros(deg.shape)
    # print(edge_index.device)
    for edge in edge_index.t():
        sim_edge = f.cosine_similarity(x[edge[0]], x[edge[1]], dim=0)
        homo_num[edge[0]] += sim_edge
        homo_num[edge[1]] += sim_edge
    homo_num = torch.tensor(homo_num, device=device)
    deg = torch.divide(homo_num, deg)   # [-1, 1]

    multi_edge_index = []
    for i in range(args.J):
        multi_edge_index.append(torch.empty(0, 2, dtype=edge_index.dtype, device=device))
    mean_sim = [0 for i in range(args.J)]
    total_sim = [ []  for i in range(args.J)]

    for edge in edge_index.t():
        #! TODO
        '''cos-sim + node-homo-sim'''
        sim = f.cosine_similarity(x[edge[0]], x[edge[1]], dim=0)
        flag = sim / 2.0 + 0.5
        src_val = deg[edge[0]]
        dst_val = deg[edge[1]]

        c_1 = - sim                                         # [-1, 1]
        c_2 = abs(dst_val - src_val)                        # [0, 2]
        alpha = args.alpha
        sim_edge = alpha*c_1 + (1-alpha)*c_2 + (alpha-1)    # [-1, 1]
        idx = check_index_homo(J, sim_edge)

        if idx < J:
            multi_edge_index[idx] = torch.cat((multi_edge_index[idx], edge.view(1, -1)), dim=0)
            total_sim[idx].append(sim_edge.detach().cpu())
        else:
            multi_edge_index[0] = torch.cat((multi_edge_index[0], edge.view(1, -1)), dim=0)
            total_sim[0].append(sim_edge.detach().cpu())

    multi_edge_index = [edge_index.t() for edge_index in multi_edge_index]
    for i in range(J):
        setattr(data, "{}_edge_index".format(i), multi_edge_index[i])
        if len(total_sim[i]) > 0:
            mean_sim[i] = sum(total_sim[i]) / len(total_sim[i])
    # print(mean_sim)
    data.mean_sim = mean_sim
    data.sub_edge_index = edge_index
    print("End multi_scale_edge_sampler")
    torch.cuda.empty_cache()
    return data


def random_sample_edges(edge_index, num_samples):
    """_summary_
    Args:
        edge_index (_type_): _description_
        num_samples (_type_): _description_
    Returns:
        _type_: _description_
    """
    if num_samples >= edge_index.size(1):
        return edge_index
    # print("begin edge random sampling!")
    indices = torch.randperm(edge_index.shape[1], device=edge_index.device)[:num_samples]    
    sampled_edge_index = edge_index[:, indices]
    # print("End edge random sampling!")
    # print(sampled_edge_index.shape)
    return sampled_edge_index