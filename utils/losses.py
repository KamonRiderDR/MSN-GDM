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



def get_loss(args, tea_out, stu_out, tea_mid, stu_mid, data, train_idx=None):
    """Loss Function Interface

    Args:
        args (_type_): _description_
        tea_out (_type_): _description_
        stu_out (_type_): _description_
        tea_mid (_type_): _description_
        stu_mid (_type_): _description_
        data (_type_): _description_
        train_idx: for OGBN dataset
    """

    out = stu_out[data.train_mask] if train_idx is None else stu_out[train_idx]
    loss_type = args.distill_type
    loss = 0.
    if loss_type == "SpecMLP":
        if "ogbn" in args.dataset:
            nll_loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        else:
            nll_loss = F.nll_loss(out, data.y[data.train_mask])
        kd_loss = KL_loss(tea_out, stu_out, T=args.tau)
        ms_optim_loss = ms_optim_loss_J(args, stu_mid, tea_mid, data)
        # loss = nll_loss \
        #         + ms_optim_loss * args.gamma2 \
        #         + kd_loss * args.gamma1
        loss = nll_loss + (ms_optim_loss + kd_loss) * args.gamma
        # loss = nll_loss + (ms_optim_loss) * args.gamma

                
    elif loss_type == "KD":
        nll_loss = F.nll_loss(out, data.y[data.train_mask])
        kd_loss = KL_loss(tea_mid, stu_mid[-1], T=args.tau) # KD Loss 
        loss = nll_loss + kd_loss * 0.6
        
    elif loss_type == "GLNN":
        loss1 = F.cross_entropy(stu_mid[-1][data.train_mask], data.y[data.train_mask])
        loss2 = F.kl_div(stu_out, tea_out, reduction="batchmean", log_target=True)
        loss2 = KL_loss(tea_out, stu_out)
        loss = loss1 * 0.4 + loss2 * 0.4
        
    elif loss_type == "FF-G2M":
        nll_loss = F.nll_loss(out, data.y[data.train_mask])
        criterion_t = nn.KLDivLoss(reduction="batchmean", log_target=True)
        criterion_t = KL_loss
        loss_l = edge_distribution_low(data.edge_index, tea_out, stu_out, criterion_t)
        loss_h = criterion_t(edge_distribution_high(data.edge_index, stu_mid[-1], tau=args.tau),
                             edge_distribution_high(data.edge_index, tea_mid, tau=args.tau),
                             T=args.tau * 2)
        loss = nll_loss + (loss_l + loss_h) * 0.4
    
    else:
        raise NameError("Unknown loss type", loss_type)
    
    return loss