'''
Description: 
Author: Rui Dong
Date: 2024-04-11 13:35:18
LastEditors: Please set LastEditors
LastEditTime: 2024-05-07 14:45:20
'''
import os
import time
import numpy as np
import seaborn
import random
import argparse
import copy
import json

import torch
import torch.nn.functional as F
import torch.nn as nn

from params import *
from utils.utils import *
from utils.losses import *
from utils.dataset import DataLoader, std_random_dataset
from model.TeacherModel import BernNet, GCN
from model.StudentModel import SpecMLP


seeds=[]
datasets = ["Cora", "Pubmed", "Computers", "Citeseer", "Photo", "Texas", "Cornell", "CS", "Actor", "DBLP", "Physics"]
uid = {}

def load_config(args):
    global datasets
    
    dict_list = []
    config_path = str("{}/config/param_dataset.json".format(args.root))
    
    for dataset_name in datasets:
        dataset = DataLoader(dataset_name)
        data = dataset[0]

        num_train_per_class = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        num_val = int(round(args.val_rate * len(data.y)))
        num_test = int(len(data.y) - num_train_per_class * dataset.num_classes - num_val)
        
        dict = {
            "dataset":                  dataset_name,
            "num_train_per_class":      num_train_per_class,
            "num_val":                  num_val,
            "num_test":                 num_test,
            "seed":                     args.seed,
            "gamma1":                   args.gamma1,
            "gamma2":                   args.gamma2
        }
        dict_list.append(dict)
    
    with open(config_path, mode='w', encoding="utf-8") as file:
        json.dump(dict_list, file, indent=1)
    file.close()


def load_training_config(args):
    config_path = str("{}/config/param_training.json".format(args.root))
    training_dict = {
        "datasets":         datasets,
        "seeds":            seeds,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "patience":         args.patience,
        "early_stopping":   args.early_stopping,
        "dropout":          args.dropout,
        "train_rate":       args.train_rate,
        "val_rate":         args.val_rate,
        
        "gamma1":           args.gamma1,
        "gamma2":           args.gamma2,
    }
    
    with open(config_path, mode='w', encoding='utf-8') as f:
        json.dump(training_dict, f, indent=1)
    f.close()


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

args = get_student_args()

if __name__ == "__main__":
    dataset = DataLoader("Physics")
    data = dataset[0]

    num_train_per_class = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    num_val = int(round(args.val_rate * len(data.y)))
    num_test = int(len(data.y) - num_train_per_class * dataset.num_classes - num_val)
    print(num_train_per_class)
    print(num_val)
    print(num_test)
