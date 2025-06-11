'''
Description: 
Author: Rui Dong
Date: 2024-04-11 16:56:32
LastEditors: Please set LastEditors
LastEditTime: 2024-06-20 16:16:43
'''

import os
import time
import numpy as np
import seaborn
import random
import argparse
import copy
from operator import methodcaller

import torch
import torch.nn.functional as F
import torch.nn as nn


from params import *
from utils.utils import *
from utils.losses import *
from utils.dataset import DataLoader, std_random_dataset
from model.TeacherModel import BernNet, GCN, GAT, ChebNet, GraphSAGE
from model.StudentModel import SpecMLP
from itertools import product

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

teacher_model_factory = {
    "BernNet":      BernNet,
    "GCN":          GCN,
    "GAT":          GAT,
    "ChebNet":      ChebNet,
    "GraphSAGE":    GraphSAGE
}

student_model_factory = {
    "SpecMLP": SpecMLP
}


cur_best_test = 0.
seeds = []
datasets = []
dist_metrics = []


# TODO
def train_epoch_minibatch(args,
                          teacher_model, student_model, 
                          loader,
                          optimizer, 
                          split_idx):
    """Train mini-batch. For OGBN dataset only.

    Args:
    Returns:
        _type_: _description_
    """
    from ogb.nodeproppred import Evaluator

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

    if args.dist_metrics == "ms_mse" or args.dist_metrics == "ms_cos":
        dist_metrics = [[] for i in range(args.J)]

    for epoch in range(args.epochs):
        #* train process
        time_begin = time.time()
        teacher_model.eval()
        student_model.train()
        # optimizer.zero_grad()

        #* start mini-batch LOOP
        total_loss = 0.
        # print(data.train_mask)
        for data in loader:
            data.to(device)
            # print(data)
            data = multi_scale_edge_sampler(args, data)
            data.to(device)
            optimizer.zero_grad()
            tea_out, tea_mid = teacher_model(data)
            stu_out, stu_mid = student_model(data)
            # print(tea_out.shape)

            # TODO
            # loss = get_loss(args, tea_out, stu_out, tea_mid, stu_mid, data)
            print(stu_out.shape)
            print(train_idx.shape)

            nll_loss = F.nll_loss(stu_out[train_idx], data.y.squeeze(1)[train_idx])
            kd_loss = KL_loss(tea_mid, stu_mid[-1], T=args.tau)
            ms_optim_loss = ms_optim_loss_J(args, stu_mid, tea_mid, data)
            loss = nll_loss + (ms_optim_loss + kd_loss) * args.gamma

            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            # total_correct += int(stu_out.argmax(dim=-1).eq(y).sum())

        duration = time.time() - time_begin
        time_run.append(duration)
        #* val && test
        evaluator = Evaluator(name=args.dataset)
        train_acc, val_acc, test_acc = test_ogbn(args, student_model, data, split_idx, evaluator)
        # early stopping
        # if val_acc > best_val_acc:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            # best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        
        #* store current best model
        if best_test_acc > cur_best_test:
            cur_best_test = best_test_acc
            save_path = str("{}/ckpt/{}".format(args.root, args.dataset))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(student_model.state_dict(), "{}/ckpt/{}/student_{}_{}.pth".format(args.root, args.dataset, args.net, args.distill_type))        
        #* print && log
        if epoch % 50 == 0:
            print("Epoch: {:03d}\ttrain acc: {:.6f} \t val acc: {:.6f} \t test acc: {:.6f} \t sim: {:.6f}".format(
                epoch, train_acc, val_acc, test_acc, 0.11111
            ))
        if patience >= args.patience:
            print("Early stop at epoch {:03d}\n".format(epoch))
            break
        
    return best_val_acc, best_test_acc    

@torch.no_grad()
def test_minibatch(args, model, test_loader):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_correct = total_samples = 0
    
    for data in test_loader:
        data.to(device)
        batch_size = data.num_sampled_nodes[0]
        out,_ = model(data)[:batch_size]
        pred = out.argmax(dim=-1)
        y = data.y[:batch_size].view(-1).to(torch.long)
        
        total_correct += int((pred==y).sum())
        total_samples += y.size(0)
    
    return total_correct / total_samples 

def train_epoch_ogbn(args, 
                     teacher_model, 
                     student_model, 
                     optimizer, 
                     data, 
                     split_idx):
    """_summary_

    Args:

    Returns:
        _type_: _description_
    """
    global cur_best_test, dist_metrics
    from ogb.nodeproppred import Evaluator

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

    # load teacher embeddings into .npy format
    # TODO
    logits_tea_npy = "{}/save/logits/teacher_{}".format(args.root, args.dataset)
    if teacher_model is not None:
        teacher_model.eval()
        tea_out, tea_mid = teacher_model(data)
        tea_out_npy = tea_out.detach().cpu().numpy()
        tea_mid_npy = tea_mid.detach().cpu().numpy()
        np.save("{}_out.npy".format(logits_tea_npy), tea_out_npy)
        np.save("{}_mid.npy".format(logits_tea_npy), tea_mid_npy)
        print("Save teacher logits successfully!")
    else:
        # load in teacher logits
        tea_out_npy = np.load("{}_out.npy".format(logits_tea_npy))
        tea_mid_npy = np.load("{}_mid.npy".format(logits_tea_npy))
        tea_out = torch.tensor(tea_out_npy).to(device)
        tea_mid = torch.tensor(tea_mid_npy).to(device)

    for epoch in range(args.epochs):
        #* train process
        time_begin = time.time()
        # teacher_model.eval()
        student_model.train()
        optimizer.zero_grad()
        
        # tea_out, tea_mid = teacher_model(data)
        stu_out, stu_mid = student_model(data)
        # if args.dist_metrics == "ms_mse" or args.dist_metrics == "ms_cos":
        #     # node_sim_dist = ms_node_sim_metrics(args, x=stu_mid[-1], data=data)
        #     node_sim_dist = ms_node_sim_metrics_ff(args, data, x=stu_mid[-1], x_tea = tea_mid)
        # else:
        #     node_sim_dist = node_sim_metrics(args, x=stu_mid[-1], edge_index=data.edge_index)
        #     # node_sim_dist = node_sim_metrics_h(args, x=stu_mid[-1], x_kd=tea_mid, edge_index=data.edge_index)

        # nll_loss = F.nll_loss(stu_out[train_idx], data.y.squeeze(1)[train_idx])
        # kd_loss = KL_loss(tea_mid, stu_mid[-1], T=args.tau)
        nll_criterion = nn.NLLLoss()
        kd_criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
        nll_loss = nll_criterion(stu_out[train_idx], data.y.squeeze(1)[train_idx])
        kd_loss = kd_criterion(stu_out, tea_out)
        # kd_loss = F.kl_div(stu_mid[-1], 
        #                    tea_mid, 
        #                    reduction='batchmean',
        #                    log_target=True)
        ms_optim_loss = ms_optim_loss_J(args, stu_mid, tea_mid, data)   # note: kv_div is for ogbn only ...
        # loss = nll_loss * (1 - args.gamma) + (ms_optim_loss + kd_loss) * args.gamma
        # loss = nll_loss * (1 - args.gamma) \
        #     + (kd_loss) * args.gamma \
        #     + ms_optim_loss * args.gamma * 0
        loss = nll_loss * 0.05 \
            + (kd_loss * args.gamma \
                + ms_optim_loss * (1 - args.gamma)) * 0.95
        
        # loss = get_loss(args, tea_out, stu_out, tea_mid, stu_mid, data)
        loss.backward()
        optimizer.step()

        duration = time.time() - time_begin
        time_run.append(duration)

        #* val && test
        evaluator = Evaluator(args.dataset)
        train_acc, val_acc, tmp_test_acc = test_ogbn(args, student_model, data, split_idx, evaluator)
        # early stopping
        #* modify
        # if val_acc > best_val_acc:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            # best_val_loss = val_loss
            patience = 0
        else:
            patience += 1

        #* store current best model
        if best_test_acc > cur_best_test:
            cur_best_test = best_test_acc
            save_path = str("{}/ckpt/{}".format(args.root, args.dataset))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # torch.save(student_model.state_dict(), "{}/ckpt/{}/student_{}_{}.pth".format(args.root, args.dataset, args.net, args.distill_type))
        
        #* print && log
        if epoch % 50 == 0:
            print("Epoch: {:03d}\ttrain acc: {:.6f} \t val acc: {:.6f} \t test acc: {:.6f} \t sim: {:.6f}".format(
                epoch, train_acc, val_acc, tmp_test_acc, 0.11111
            ))
        if patience >= args.patience:
            print("Early stop at epoch {:03d}\n".format(epoch))
            break
        torch.cuda.empty_cache()    
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

def train_epoch(args, 
                teacher_model, 
                student_model, 
                optimizer, 
                data):
    """ Train student model for 1 time * epochs 

    Args:
        args 
        teacher_model 
        student_model 
        optimizer 
        data: `PYG` Data
        
    Return:
        best_val_acc, best_test_acc
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
    
    if args.dist_metrics == "ms_mse" or args.dist_metrics == "ms_cos":
        dist_metrics = [[] for i in range(args.J)]
    
    for epoch in range(args.epochs):
        #* train process
        time_begin = time.time()
        teacher_model.eval()
        student_model.train()
        optimizer.zero_grad()

        tea_out, tea_mid = teacher_model(data)
        stu_out, stu_mid = student_model(data)
        if args.dist_metrics == "ms_mse" or args.dist_metrics == "ms_cos":
            # node_sim_dist = ms_node_sim_metrics(args, x=stu_mid[-1], data=data)
            node_sim_dist = ms_node_sim_metrics_ff(args, data, x=stu_mid[-1], x_tea = tea_mid)
        else:
            node_sim_dist = node_sim_metrics(args, x=stu_mid[-1], edge_index=data.edge_index)
            # node_sim_dist = node_sim_metrics_h(args, x=stu_mid[-1], x_kd=tea_mid, edge_index=data.edge_index)

        loss = get_loss(args, tea_out, stu_out, tea_mid, stu_mid, data)

        loss.backward()
        optimizer.step()

        duration = time.time() - time_begin
        time_run.append(duration)

        #* val && test
        [train_acc, val_acc, tmp_test_acc], preds, \
        [train_loss, val_loss, tmp_test_loss] = test(student_model, data)
        # if val_acc > best_val_acc:
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1

        #* store current best model
        if best_test_acc > cur_best_test:
            cur_best_test = best_test_acc
            save_path = str("{}/ckpt/{}".format(args.root, args.dataset))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(student_model.state_dict(), "{}/ckpt/{}/student_{}_{}.pth".format(args.root, args.dataset, args.net, args.distill_type))

        #* print && log
        if epoch % 2 == 0:
            # dist_metrics.append(node_sim_dist.detach().cpu())
            if args.dist_metrics == "ms_mse" or args.dist_metrics == "ms_cos":
                for i in range(len(node_sim_dist)):
                    dist_metrics[i].append(node_sim_dist[i].detach().cpu())
            else:
                    dist_metrics.append(node_sim_dist.detach().cpu())

        if epoch % 50 == 0:
            print("Epoch: {:03d}\ttrain loss: {:.6f} \t val acc: {:.6f} \t test acc: {:.6f} \t sim: {:.6f}".format(
                epoch, train_loss, val_acc, tmp_test_acc, 0.11111
            ))
        if patience >= args.patience:
            print("Early stop at epoch {:03d}\n".format(epoch))
            break
    
    # save dist-metrics
    if args.dist_metrics == "ms_mse" or args.dist_metrics == "ms_cos":
        np.save("{}/save/dist_metrics/ms_{}_{}_dist.npy".format(args.root, args.distill_type, args.dataset), dist_metrics)
        if args.distill_type == "SpecMLP":
            # ms_plot_curve(args)
            # ms_plot_curve_J(args)
            ms_plot_curve_ff(args, data)
            print("Fuck you")
    else:
        np.save("{}/save/dist_metrics/{}_{}_dist.npy".format(args.root, args.distill_type, args.dataset), np.array(dist_metrics))
        # np.save("{}/save/dist_metrics/{}_{}_dist_h.npy".format(args.root, args.distill_type, args.dataset), np.array(dist_metrics))
        if args.distill_type == "SpecMLP":
            plot_dist_curve(args)
    dist_metrics.clear()
    
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
        args 
    """
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    
    dataset = DataLoader(args.dataset)
    data = dataset[0]
    num_train_per_class = int( round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int( round(args.val_rate * len(data.y)) )
    
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    
    test_accs = []

    for i in range(args.runs):
        print("Running {:03d} times".format(i+1))
        args.seed = seeds[i]
        teacher_model = teacher_model_factory[args.net](args)
        student_model = student_model_factory[args.student_model](args)
        teacher_model.load_state_dict(torch.load("{}/ckpt/{}/teacher_{}.pth".format(args.root, args.dataset, args.net)))

        #* modify here, cause data structure does not change
        data = random_planetoid_splits(data, dataset.num_classes, num_train_per_class, val_lb, args.seed)        
        teacher_model.to(device)
        student_model.to(device)
        data.to(device)    

        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc, best_test_acc = train_epoch(args, teacher_model, student_model, optimizer, data)

        print("{:03d}_run\t val_acc:{:.6f} test_acc:{:.6f}".format(i+1, best_val_acc, best_test_acc))
        test_accs.append(best_test_acc)

    test_acc_mean = np.mean(test_accs) * 100
    test_acc_std = np.std(test_accs) * 100
    
    print("{:3d} times results: {:.2f} ± {:.2f}".format(i+1, test_acc_mean, test_acc_std))
    current_time = time.strftime("%m/%d %H:%M", time.localtime(time.time()))
    results_path = str("{}/logs/results/student/{}_{}_{}_{}.txt".format(args.root, args.net, args.student_model, args.dataset, args.device))
    with open(results_path, 'a+') as f:
        f.write("{}\t{:3d} times results:\t{:.2f} ± {:.2f}\n".format(current_time, i+1, test_acc_mean, test_acc_std))
    f.close()


def train_std(args):
    """Train a complete process for [run] TIMES. Use `PYG` official random split method.

    Args:
        args:

    """
    
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    
    if "ogbn" not in args.dataset:
        json_path = str("{}/config/param_dataset.json".format(args.root))
        param_dict = load_dataset_param(args, json_path)
        num_train_per_class = param_dict["num_train_per_class"]
        num_val = param_dict["num_val"]
        num_test = param_dict["num_test"]
        seed = param_dict["seed"]
    # args.J = param_dict["J"]
    # args.gamma1 = param_dict["gamma1"]
    # args.gamma2 = param_dict["gamma2"]
    
    # dataset = DataLoader(args.dataset)
    # data = dataset[0]
    # num_train_per_class = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    # num_val = int(round(args.val_rate * len(data.y)))
    # num_test = int(len(data.y) - num_train_per_class * dataset.num_classes - num_val)

    test_accs = []
    results_path = str("{}/logs/results/student/{}_{}_{}_{}.txt".format(args.root, args.net, args.student_model, args.dataset, args.device))

    for i in range(args.runs):
        print("Running {:03d} times".format(i+1))
        args.seed = 666
        setup_seed(args.seed)
        
        if "ogbn" not in args.dataset:
            dataset = std_random_dataset(args.dataset, num_train_per_class, num_val, num_test)
        else:
            print("This is ogbn dataset")
            dataset = std_random_dataset(args.dataset)
            split_idx =  dataset.get_idx_split()

        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features    

        # dataset = dataset_node_sampler(args, dataset)
        # preprocess for toy datasets
        if args.distill_type == "SpecMLP" and "ogbn" not in args.dataset:
            dataset = multi_scale_node_sampler(args, dataset)
        data = dataset[0]
        if "ogbn" in args.dataset:
            # print("Hey Jude")
            data.adj_t = data.adj_t.to_symmetric()
            row, col,_ = data.adj_t.t().coo()
            data.edge_index = torch.stack([row, col], axis=0)
            data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

        teacher_model = teacher_model_factory[args.net](args)
        student_model = student_model_factory[args.student_model](args)
        teacher_model.load_state_dict(torch.load("{}/ckpt/{}/teacher_{}.pth".format(args.root, args.dataset, args.net)))

        teacher_model.to(device)
        student_model.to(device)
        if args.train != "mini-batch":
            data.to(device)    

        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if args.train == "mini-batch":
            loader = ogbn_dataloader(args, data)
            best_val_acc, best_test_acc = train_epoch_minibatch(args, 
                                                                teacher_model, student_model,
                                                                loader,
                                                                optimizer,
                                                                split_idx)
        elif "ogbn" in args.dataset:
            data = multi_scale_edge_sampler(args, data)
            best_val_acc, best_test_acc = train_epoch_ogbn(args, 
                                                           teacher_model=None, 
                                                           student_model=student_model, 
                                                           optimizer=optimizer, 
                                                           data=data, 
                                                           split_idx=split_idx)
        else:
            best_val_acc, best_test_acc = train_epoch(args, 
                                                      teacher_model, student_model, 
                                                      optimizer, 
                                                      data)
        
        print("{:03d}_run\t val_acc:{:.6f} test_acc:{:.6f}".format(i+1, best_val_acc, best_test_acc))
        test_accs.append(best_test_acc)

        current_time = time.strftime("%m/%d %H:%M", time.localtime(time.time()))
        with open(results_path, 'a+') as f:
            f.write("{:.2f}\t".format(best_test_acc * 100))
        f.close()
        torch.cuda.empty_cache()
    
    test_acc_mean = np.mean(test_accs) * 100
    test_acc_std = np.std(test_accs) * 100
    print("{:3d} times results: {:.2f} ± {:.2f}".format(i+1, test_acc_mean, test_acc_std))
    current_time = time.strftime("%m/%d %H:%M", time.localtime(time.time()))
    with open(results_path, 'a+') as f:
        f.write("\n{}\t{:3d} times results:\t{:.2f} ± {:.2f}\n".format(current_time, args.runs, test_acc_mean, test_acc_std))
        if args.distill_type == "SpecMLP":
            f.write("multi-set: {}\n".format(data.mean_sim))
    f.close()


def run(args, type=None):
    global cur_best_test
    results_path = str("{}/logs/results/student/{}_{}_{}_{}.txt".format(args.root, args.net, args.student_model, args.dataset, args.device))
    with open(results_path, 'a+') as f:
        f.write("----------------------------------------------------\n")
    f.close()
    for i in range(args.times):
        cur_best_test = 0.
        if type is not None:
            train_std(args)
        else:
            train(args)
        

def init_training_params(args):
    global datasets, seeds
    
    config_path = str("{}/config/param_training.json".format(args.root))
    with open(config_path, mode="r", encoding="utf-8") as f:
        training_dict = json.load(f)
        datasets = training_dict["datasets"]
        seeds = training_dict["seeds"]

        args.gamma1 = training_dict["gamma1"]
        args.gamma2 = training_dict["gamma2"]
        # args.lr = training_dict["lr"]
        # args.epochs = training_dict["epochs"]
        args.J = training_dict["J"]
    f.close()

def grid_search(args):
    gammas = [0.3, 0.4, 0.5, 0.6, 0.7]
    J_s = [2, 3, 4, 5, 6, 7]
    alpha_s = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # For OGBN
    # dropouts = [0.0, 0.05, 0.1, 0.15, 0.2]
    # gammas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    # gammas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    # gammas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # gammas = [0.7]
    # J_s = [3, 4, 5, 6]
    # alpha_s = [0.2, 0.4, 0.6, 0.8]

    for gamma_, alpha_, J_ in product(gammas, alpha_s, J_s):
        args.gamma = gamma_
        args.J = J_
        args.alpha = alpha_
        
        results_path = str("{}/logs/results/student/{}_{}_{}_{}.txt".format(args.root, args.net, args.student_model, args.dataset, args.device))
        with open(results_path, 'a+') as f:
            f.write("\n------------gamma: {}---------------------alpha:{} ---------------J: {} -------------------DT: {}-----------------\n".format(
                args.gamma, args.alpha, args.J, args.distill_type))
        f.close()
        # ablations = ["KD", "GLNN", "FF-G2M"]
        # for name in ablations:
        #     args.distill_type = name
        #     run(args, type="std")
        # args.distill_type = "SpecMLP"
        run(args, type="std")


if __name__ == '__main__':
    args = get_student_args()
    init_training_params(args)
    grid_search(args)
    # args.alpha = 0.3
    # args.gamma = 1.0
    # args.J = 4
    # run(args, type="std")
    # ablations = ["KD", "GLNN", "FF-G2M"]
    # for name in ablations:
    #     args.distill_type = name
    #     run(args, type="std")
    # args.distill_type = "SpecMLP"
    # run(args, type="std")    
    # distill_s = ["FF-G2M", "GLNN", "KD"]
    # # distill_s = ["GLNN", "KD"]
    # for distill_ in distill_s:
    #     args.distill_type = distill_
    #     run(args, type="std")

