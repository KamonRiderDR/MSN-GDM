import os
import time
import numpy as np
import seaborn
import random
import argparse
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import NeighborSampler

from ogb.nodeproppred import Evaluator

from params import *

from utils.utils import *
from utils.dataset import DataLoader, std_random_dataset
from model.TeacherModel import BernNet, GCN, GAT, ChebNet, GraphSAGE
from model.StudentModel import SpecMLP

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


teacher_model_factory = {
    "BernNet":      BernNet,
    "GCN":          GCN,
    "GAT":          GAT,
    "ChebNet":      ChebNet,
    "GraphSAGE":    GraphSAGE,
    "SpecMLP":      SpecMLP
}

cur_best_test = 0.

seeds=[]
datasets = ["Cora", "Pubmed", "Computers", "Citeseer", "Photo", "Texas", "Cornell", "CS"]
dist_metrics = []


def train_epoch_ogbn(args, model, optimizer, data, split_idx):
    """train for all the epoches during ONE TIME.
    This is a copy from 
        https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py#L79
    
    Args:

    Return:

    """
    global cur_best_test, dist_metrics
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    train_idx = split_idx['train'].to(device)
    evaluator = Evaluator(name=args.dataset)

    best_val_loss = float('inf')
    best_val_acc = 0.
    best_test_acc = 0.
    time_run = []
    patience = 0
    best_weights = None
    
    val_loss_history = []
    val_acc_history = []

    # #* TEMP. DELETE THIS IN THE END        
    for epoch in range(args.epochs):
        #* train process
        time_begin = time.time()
        model.train()
        optimizer.zero_grad()
        
        out, mid = model(data) 
        out = out[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        
        loss.backward()
        optimizer.step()

        duration = time.time() - time_begin
        time_run.append(duration)

        #* val && test
        [train_acc, val_acc, tmp_test_acc] = test_ogbn(args, model, data, split_idx, evaluator)
        #* early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            # best_val_loss = val_acc
            patience = 0
        else:
            patience += 1
        # store current best model
        if best_test_acc > cur_best_test:
            cur_best_test = best_test_acc
            save_path = str("{}/ckpt/{}".format(args.root, args.dataset))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), "{}/ckpt/{}/teacher_{}.pth".format(args.root, args.dataset, args.net))

        if epoch % 50 == 0:
            print("Epoch: {:03d}\ttrain acc: {:.6f} \t val acc: {:.6f} \t test acc: {:.6f}".format(
                epoch, train_acc, val_acc, tmp_test_acc
            ))
        if patience >= args.patience:
            print("Early stop at epoch {:03d}\n".format(epoch))
            break

    return best_val_acc, best_test_acc


@torch.no_grad()
def test_ogbn(args, model, data, split_idx, evaluator):
    """This is a copy from 
        https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py#L92
        
    Args:
    Returns:
        [train_acc, valid_acc, test_acc]
    """
    model.eval()
    out,_ = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc



def train_epoch(args, model, optimizer, data):
    """train for all the epoches during ONE TIME
    
    Args:
        args (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        data (_type_): _description_
        train_idx: None for non-ogbn dataset
    Return:

    """
    global cur_best_test, dist_metrics
    
    best_val_loss = float('inf')
    best_val_acc = 0.
    best_test_acc = 0.
    time_run = []
    patience = 0
    best_weights = None
    
    val_loss_history = []
    val_acc_history = []

    # #* TEMP. DELETE THIS IN THE END    
    # device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    # args.hidden = 256
    # teacher_model = GCN(args)
    # teacher_model.load_state_dict(torch.load("{}/ckpt/{}/teacher_GCN.pth".format(args.root, args.dataset)))
    # teacher_model.to(device)
    # tea_out, tea_mid = teacher_model(data)
    # args.hidden = 64
    
    for epoch in range(args.epochs):
        #* train process
        time_begin = time.time()
        model.train()
        optimizer.zero_grad()
        out, mid = model(data) 
        out = out[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # node_sim_dist = node_sim_metrics(args, x=mid, edge_index=data.edge_index)        
        # node_sim_dist = node_sim_metrics(args, x=mid[-1], edge_index=data.edge_index)
        # node_sim_dist = node_sim_metrics_h(args, x=mid[-1], x_kd=tea_mid, edge_index=data.edge_index)

        duration = time.time() - time_begin
        time_run.append(duration)
    
        #* val && test
        [train_acc, val_acc, tmp_test_acc], preds, \
        [train_loss, val_loss, tmp_test_loss] = test(model, data)
        #* early stopping
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        # store current best model
        if best_test_acc > cur_best_test:
            cur_best_test = best_test_acc
            save_path = str("{}/ckpt/{}".format(args.root, args.dataset))
            # best_weights = copy.deepcopy(model.state_dict())
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), "{}/ckpt/{}/teacher_{}.pth".format(args.root, args.dataset, args.net))

        # if epoch % 2 == 0:
        #     dist_metrics.append(node_sim_dist.detach().cpu())
        if epoch % 50 == 0:
            print("Epoch: {:03d}\ttrain loss: {:.6f} \t val acc: {:.6f} \t test acc: {:.6f}".format(
                epoch, train_loss, val_acc, tmp_test_acc
            ))
        if patience >= args.patience:
            print("Early stop at epoch {:03d}\n".format(epoch))
            break
        
    # np.save("{}/save/dist_metrics/tea_{}_{}_dist.npy".format(args.root, args.net, args.dataset), np.array(dist_metrics))
    # np.save("{}/save/dist_metrics/tea_{}_{}_dist_h.npy".format(args.root, args.net, args.dataset), np.array(dist_metrics))
    # dist_metrics = []

    return best_val_acc, best_test_acc

def test(model, data):
    """Test teacher model

    Args:
        args: args
        model: teacher model
        data: `PYG` data
    
    Return:
        accs, preds, losses of [train, val, test]
    """
    model.eval()
    logits, _ = model(data)
    accs = []
    preds = []
    losses = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        
        out_, _ = model(data) 
        loss = F.nll_loss(out_[mask], data.y[mask])
        
        accs.append(acc)
        preds.append(pred.detach().cpu())
        losses.append(loss.detach().cpu())
    return accs, preds, losses


def train(args):
    """Train for [run] TIMES

    Args:
        args (_type_): _description_
        dataset (_type_): _description_
        model (_type_): _description_
        percls_trn (_type_): _description_
        val_lb (_type_): _description_
    """
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    
    dataset = DataLoader(args.dataset)
    data = dataset[0]
    percls_trn = int( round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int( round(args.val_rate * len(data.y)) )
    
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    
    test_accs = []

    for i in range(args.runs):
        print("Running {:03d} times".format(i+1))
        args.seed = seeds[i]
        model = teacher_model_factory[args.net](args)
        #* modify here, cause data structure does not change
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)        
        model.to(device)
        data.to(device)    
        
        if args.net =='BernNet':
            optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc, best_test_acc = train_epoch(args, model, optimizer, data)
        
        print("{:03d}_run\t val_acc:{:.6f} test_acc:{:.6f}".format(i+1, best_val_acc, best_test_acc))
        test_accs.append(best_test_acc)
    
    test_acc_mean = np.mean(test_accs) * 100
    test_acc_std = np.std(test_accs) * 100
    
    print("{:3d} times results: {:.2f} ± {:.2f}".format(i+1, test_acc_mean, test_acc_std))
    current_time = time.strftime("%m/%d %H:%M", time.localtime(time.time()))
    results_path = str("{}/logs/results/teacher/{}_{}.txt".format(args.root, args.dataset, args.net))
    with open(results_path, 'a+') as f:
        f.write("{}\t{:3d} times results:\t{:.2f} ± {:.2f}\n".format(current_time, i+1, test_acc_mean, test_acc_std))
    f.close()


def train_std(args):
    """Train for [run] TIMES

    Args:
        args (_type_): _description_
        dataset (_type_): _description_
        model (_type_): _description_
        percls_trn (_type_): _description_
        val_lb (_type_): _description_
    """
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # dataset = DataLoader(args.dataset)
    # data = dataset[0]
    # num_train_per_class = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    # num_val = int(round(args.val_rate * len(data.y)))
    # num_test = int(len(data.y) - num_train_per_class * dataset.num_classes - num_val)
    
    json_path = str("{}/config/param_dataset.json".format(args.root))
    if "ogbn" not in args.dataset:
        param_dict = load_dataset_param(args, json_path)
        num_train_per_class = param_dict["num_train_per_class"]
        num_val = param_dict["num_val"]
        num_test = param_dict["num_test"]

    test_accs = []
    results_path = str("{}/logs/results/teacher/{}_{}.txt".format(args.root, args.dataset, args.net))

    for i in range(args.runs):
        print("Running {:03d} times".format(i+1))
        # args.seed = param_dict["seed"]
        args.seed = 666
        # args.seed = seeds[i]
        setup_seed(args.seed)
        
        dataset = std_random_dataset(args.dataset)
        # dataset = std_random_dataset(args.dataset, num_train_per_class, num_val, num_test)            
        data = dataset[0]    
        if "ogbn" in args.dataset:
            data.adj_t = data.adj_t.to_symmetric()

        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features        

        model = teacher_model_factory[args.net](args)
        # data = random_planetoid_splits(data, args.num_classes)
        # data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)        
        model.to(device)
        data.to(device)    
        
        if args.net =='BernNet':
            optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if "ogbn" in args.dataset:
            split_idx = dataset.get_idx_split()
            best_val_acc, best_test_acc = train_epoch_ogbn(args, model, optimizer, data, split_idx)
        else:
            best_val_acc, best_test_acc = train_epoch(args, model, optimizer, data)
        
        print("{:03d}_run\t val_acc:{:.6f} test_acc:{:.6f}".format(i+1, best_val_acc, best_test_acc))
        test_accs.append(best_test_acc)
        current_time = time.strftime("%m/%d %H:%M", time.localtime(time.time()))
        with open(results_path, 'a+') as file:
            file.write("{} \t{:3d} runs results:\t{:.2f}\n".format(current_time, i+1, best_test_acc * 100))
        file.close()
    
    test_acc_mean = np.mean(test_accs) * 100
    test_acc_std = np.std(test_accs) * 100
    
    print("{:3d} times results: {:.2f} ± {:.2f}".format(i+1, test_acc_mean, test_acc_std))
    current_time = time.strftime("%m/%d %H:%M", time.localtime(time.time()))
    with open(results_path, 'a+') as f:
        f.write("{}\t{:3d} times results:\t{:.2f} ± {:.2f}\n".format(current_time, args.runs, test_acc_mean, test_acc_std))
    f.close()



def run(args, type=None):
    global cur_best_test
    cur_best_test = 0.
    results_path = str("{}/logs/results/teacher/{}_{}.txt".format(args.root, args.dataset, args.net))
    with open(results_path, 'a+') as f:
        f.write("----------------------------------------------------\n")
    f.close()
    for i in range(args.times):
        if type is not None:
            train_std(args)
        else:
            train(args)
        

if __name__ == "__main__":
    args = get_student_args()
    run(args, type="std")
    # for dataset in datasets:
    #     args.dataset = dataset
    #     run(args)