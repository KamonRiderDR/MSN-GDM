'''
Description: 
Author: Rui Dong
Date: 2024-04-11 18:17:28
LastEditors: Please set LastEditors
LastEditTime: 2024-06-19 16:59:07
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



def KL_loss(teacher_output, student_output, T=3):   # T=10
    p_s = F.log_softmax(student_output / T, dim=1)
    p_t = F.softmax(teacher_output / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='sum') * (T ** 2) / student_output.shape[0]

    return loss


def KD_loss(teacher_output, student_output, T=3):
    p_s = F.log_softmax(student_output / T, dim=1)
    p_t = F.softmax(teacher_output / T, dim=1)
    loss = F.kl_div(p_s / T, p_t / T, reduction='batchmean')

    return loss

def edge_distribution_high(edge_idx, feats, tau):
    src = edge_idx[0]
    dst = edge_idx[1]

    feats_abs = torch.abs(feats[src] - feats[dst])
    e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

    return e_softmax

def edge_distribution_low(edge_idx, feats, out, criterion_t):
    src = edge_idx[0]
    dst = edge_idx[1]

    loss = criterion_t(feats[src], out[dst])

    return loss

def edge_distribution_high_coffi(edge_idx, feats, coffi, tau=1):

    src = edge_idx[0]
    dst = edge_idx[1]

    feats_abs = torch.abs(feats[src] - feats[dst]) * coffi
    e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

    return e_softmax



'''Below are Multi-Frequency-Distillation loss function'''


# TODO
def MFD_loss_J(args, tea_out, stu_out, data, multi_edge_index=None):
    loss = 0.
    J = args.J
    if multi_edge_index == None:
        multi_edge_index = []
        for i in range(J):
            multi_edge_index.append(getattr(data, "{}_edge_index".format(i)))
    criterion = KL_loss
    criterion_1 = nn.MSELoss()
    mean_sim = data.mean_sim
    
    tau = args.tau    
    for i in range(J):
        tau += 0.5
        if J == 1:
            coffi == 1.
        else:
            coffi = 1 - 2 * i / J
            coffi = mean_sim[i]
        # if coffi >= 0:
        if coffi <= 0:
            # loss += criterion(tea_out[multi_edge_index[i][1]], 
            #                   stu_out[multi_edge_index[i][0]], 
            #                   T=tau) * coffi
            loss += criterion_1(tea_out[multi_edge_index[i][1]], 
                                stu_out[multi_edge_index[i][0]]) * abs(coffi)

        if coffi >= 0:
        # if coffi <= 0:
            # loss += criterion(edge_distribution_high_coffi(multi_edge_index[i], tea_out, -coffi),
            #                   edge_distribution_high_coffi(multi_edge_index[i], stu_out, -coffi), T=tau) * (-coffi)
            loss += criterion_1(edge_distribution_high_coffi(multi_edge_index[i], tea_out, abs(coffi)),
                                edge_distribution_high_coffi(multi_edge_index[i], stu_out, abs(coffi))) * abs(coffi)

    torch.cuda.empty_cache()
    return loss / args.J


def MFD_inter_loss_J(args, stu_mid, data, multi_edge_index, i):
    """ Small k -> coffi(k) > 0 -> close

    Args:
        args:
        stu_mid:
        data:
        multi_edge_index:
        i:

    Returns:
        _type_: _description_
    """
    criterion = nn.MSELoss()
    J = args.J
    loss = 0.
    mean_sim = data.mean_sim
    
    for k in range(J):
        if J == 1:
            coffi = 1.
        else:
            coffi = 1 - 2 * k / J
            coffi = mean_sim[i]
            # print(coffi)
        # if coffi >= 0:
        if coffi <= 0:
            loss += criterion(stu_mid[i][multi_edge_index[k][0]] * abs(coffi), 
                              stu_mid[i+1][multi_edge_index[k][1]] * abs(coffi))

        else:
            loss += criterion(edge_distribution_high_coffi(multi_edge_index[k], stu_mid[i], abs(coffi), tau=args.tau),\
                              edge_distribution_high_coffi(multi_edge_index[k], stu_mid[i+1], abs(coffi), tau=args.tau))
        torch.cuda.empty_cache()
    return loss / args.J


# TODO
def ms_optim_loss_J(args, stu_mid, tea_mid, data):
    edge_index = data.edge_index if "ogbn" not in args.dataset else data.sub_edge_index
    J =args.J
    multi_edge_index = []
    for i in range(J):
        multi_edge_index.append(getattr(data, "{}_edge_index".format(i)))
    
    criterion = nn.MSELoss()
    src = edge_index[0]
    dst = edge_index[1]
    loss = 0.
    for i in range(len(stu_mid) - 2):
        loss_intra = criterion(stu_mid[i][src], stu_mid[i][dst]) / edge_index.shape[0]
        
        loss_inter_node = criterion(stu_mid[i], stu_mid[i+1]) / stu_mid[i].shape[0]
        loss_inter = loss_inter_node + MFD_inter_loss_J(args, stu_mid, data, multi_edge_index, i)
        # loss_inter = MFD_inter_loss_J(args, stu_mid, data, multi_edge_index, i)

        loss += (loss_intra + loss_inter)
        # loss += loss_inter
        # loss += loss_intra
        torch.cuda.empty_cache()    
    loss += MFD_loss_J(args, tea_mid, stu_mid[-1], data, multi_edge_index)
    torch.cuda.empty_cache()
    return loss / ( len(stu_mid) )
