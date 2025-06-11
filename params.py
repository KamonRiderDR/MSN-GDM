'''
Description: 
Author: Rui Dong
Date: 2024-04-11 14:16:42
LastEditors: Please set LastEditors
LastEditTime: 2024-06-19 16:31:07
'''
import argparse

'''val early stop dataset: [Citeseer, Computers, CS, Physics]'''

def get_teacher_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')  # 0.001
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=128, help='hidden units.')    # 256 for SAGE-Citeseer, Computers, CS
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--root', type=str, default="/home/dongrui/code/mm-gmlp/SpecMLP")
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--patience', type=int, default=200, help='early stopping.')
    parser.add_argument('--n_layers', type=int, default=3, help='layers of teacher model, [GCN, GAT, GraphSAGE]')
    parser.add_argument('--stu_layers', type=int, default=3, help='layers of teacher model, [GCN, GAT, GraphSAGE]')
    
    parser.add_argument('--num_features', type=int, help="input_dim of graph")
    parser.add_argument('--num_classes', type=int, help="output dim of graph")

    parser.add_argument('--train_rate', type=float, default=0.2, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, default='BernNet')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    parser.add_argument('--times', type=int, default=1)
    
    args = parser.parse_args()
    return args

def get_student_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=3000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')   # 0.001      
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden units.') # 64, 256 for SAGE-Citeseer, Computers, CS
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout for neural networks.')
    parser.add_argument('--root', type=str, default="/home/dongrui/code/mm-gmlp/SpecMLP")
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--patience', type=int, default=200, help='early stopping.')
    parser.add_argument('--n_layers', type=int, default=3, help='layers of teacher model, [GCN, GAT, GraphSAGE]')
    parser.add_argument('--stu_layers', type=int, default=3, help='layers of student model, [GCN, GAT, GraphSAGE]')
    parser.add_argument('--J', type=int, default=3)
    parser.add_argument('--train', type=str, default='mini-batch', help='mini-batch')
    parser.add_argument('--fan_out', type=int, default=10, help='number of neighbors in each layer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of mini-batch')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers')

    parser.add_argument('--num_features', type=int, help="input_dim of graph")
    parser.add_argument('--num_classes', type=int, help="output dim of graph")
    parser.add_argument('--stu_hidden', type=int, default=64, help='hidden units.')

    parser.add_argument('--train_rate', type=float, default=0.2, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')
    parser.add_argument('--tau', type=float, default=1.5, help="temperature")

    parser.add_argument('--distill_type', type=str, default="SpecMLP")
    parser.add_argument('--student_model', type=str, default='SpecMLP')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, default='BernNet')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    parser.add_argument('--times', type=int, default=1)
    #* hypermeter
    parser.add_argument('--gamma1', type=float, default=0.3)
    parser.add_argument("--gamma2", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--dist_metrics", type=str, default="cos")  # cos default, [cos, mse, ms_cos, ms_mse]
    
    args = parser.parse_args()
    return args

# args = get_student_args()
# print(getattr(args, 'J'))