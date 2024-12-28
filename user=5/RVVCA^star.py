import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import logging
import math
from test_result import generateceshi
def load_data(dir):
    data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
            np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]
    return torch.tensor(np.array(data)).to(torch.float32)

def generate_all_deterministic_alloc(n_agents, m_items,device,unit_demand = False) -> torch.tensor: # n buyers, m items -> alloc (n+1, m)
    alloc_num = n_agents ** (m_items)
    def gen(t, i, j):
        x = n_agents ** (m_items - 1 - j)
        return np.where((t // x) % n_agents == i, 1.0, 0.0)
    alloc = np.fromfunction(gen, (alloc_num, n_agents, m_items))
    return torch.tensor(alloc).to(torch.float32).to(device).requires_grad_(True)
def generate_revenue_loss(allocs,virtual_utility,temp_w,allocs_two_value,B,allocs_two_index,n_agents,m_items,device):
    util_from_items = ((allocs * virtual_utility.unsqueeze(1)).sum(axis=-1))*temp_w+allocs_two_value
    # (bs,1,n)       (bs, ms, n)
    per_agent_welfare = util_from_items
    total_welfare = per_agent_welfare.sum(axis=-1)
    alloc_choice_ind = torch.argmax(total_welfare, -1)

    chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in
                                      range(B)]
    chosen_alloc_welfare_per_agent = torch.stack \
        (chosen_alloc_welfare_per_agent)


    payments = []
    mask = torch.ones((n + 1, m), device=device)
    zero_mask = torch.zeros(m, device=device)
    one_mask = torch.ones(m, device=device)
    for i in range(n):
        mask[i] = zero_mask

        removed_i_welfare = ((allocs * mask.reshape(1, 1, n + 1, m) * virtual_utility.unsqueeze(1)).sum(
            axis=-1)) * temp_w + allocs_two_value
        total_removed_welfare = removed_i_welfare.sum(-1)
        removed_alloc_choice_ind = torch.argmax(total_removed_welfare, -1)
        removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in
                                  range(B)]
        removed_chosen_welfare = torch.stack(removed_chosen_welfare)

        payments.append(
            (1.0 / temp_w[:,0, i])
            *
            (
                    (chosen_alloc_welfare_per_agent.sum(1)) - (removed_chosen_welfare)

            )

        )
        mask[i] = one_mask

    payments = torch.stack(payments)


    revenue_loss = payments.sum(0).mean().item()
    del payments
    del mask
    del allocs_two_value

    return revenue_loss

def generate_result(allocs,input_bids,input_values,virtual_utility,w,lamb,B,two_index,device):
    temp_w = w.reshape(1, 1, n + 1).repeat(B, (n + 1) ** m, 1)

    allocs_two_index = (allocs * two_index.reshape(1, 1, 1, m)).sum(-1)  # B,t,n+1

    allocs_two_value = torch.zeros(((n + 1) ** m, n + 1), device=device)
    for j in range((n + 1) ** m):
        for k in range(n + 1):
            allocs_two_value[j, k] = lamb[k][allocs_two_index[0, j, k].to(torch.long)]
    allocs_two_value = allocs_two_value.unsqueeze(0).repeat(B, 1, 1)

    util_from_items = ((allocs * virtual_utility.unsqueeze(1)).sum(axis=-1))*temp_w+allocs_two_value

    per_agent_welfare = util_from_items
    total_welfare = per_agent_welfare.sum(axis=-1)
    alloc_choice_ind = torch.argmax(total_welfare, -1)

    item_allocation = [allocs[i, alloc_choice_ind[i], ...] for i in range(B)]
    item_allocation = torch.stack(item_allocation)

    chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in
                                      range(B)]
    chosen_alloc_welfare_per_agent = torch.stack \
        (chosen_alloc_welfare_per_agent)

    payments = []
    mask = torch.ones((n + 1, m), device=device)
    zero_mask = torch.zeros(m, device=device)
    one_mask = torch.ones(m, device=device)
    for i in range(n):
        mask[i] = zero_mask
        # (bs,ms,n)         (1,1,n)
        removed_i_welfare = ((allocs * mask.reshape(1, 1, n + 1, m) * virtual_utility.unsqueeze(1)).sum(
            axis=-1)) * temp_w + allocs_two_value
        total_removed_welfare = removed_i_welfare.sum(-1)
        removed_alloc_choice_ind = torch.argmax(total_removed_welfare, -1)
        removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in
                                  range(B)]
        removed_chosen_welfare = torch.stack(removed_chosen_welfare)

        payments.append(
            (1.0 / temp_w[:, 0, i])
            *
            (
                    (chosen_alloc_welfare_per_agent.sum(1)) - (removed_chosen_welfare)
            )
            # (bs,n,m)*(bs,n,m)
        )
        mask[i] = one_mask

    payments = torch.stack(payments)

    payments = payments + (input_bids * item_allocation[:,:-1,:]).sum(-1).permute(1, 0)
    utility = (input_values * item_allocation[:,:-1,:]).sum(-1).permute(1, 0) - payments

    del mask
    del allocs_two_value

    return utility.sum(0).mean(), payments.sum(0).mean(), (input_bids * item_allocation[:,:-1,:]).sum(-1).permute(1,0).sum(0).mean()

def generate_final_revenue_loss(allocs,w,B,lamb,virtual_utility,two_index):
    temp_w = w.reshape(1, 1, n + 1).repeat(B, (n + 1) ** m, 1)



    allocs_two_index = (allocs * two_index.reshape(1, 1, 1, m)).sum(-1)  # B,t,n+1


    allocs_two_value = torch.zeros(((n + 1) ** m, n + 1), device=device)
    for j in range((n + 1) ** m):
        for k in range(n + 1):
            allocs_two_value[j, k] = lamb[k][allocs_two_index[0, j, k].to(torch.long)]
    allocs_two_value = allocs_two_value.unsqueeze(0).repeat(B, 1, 1)

    revenue_loss = generate_revenue_loss(allocs, virtual_utility, temp_w, allocs_two_value, B, allocs_two_index, n, m,
                                         device)
    del allocs_two_index
    del allocs_two_value

    return revenue_loss

def gradCalculate(allocs,w, B, lamb, virtual_utility,two_index,device):
    grad_w0 = torch.zeros(n + 1, device=device).to(torch.float32)
    grad_b0 = torch.zeros((n + 1, 2 ** m), device=device).to(torch.float32)
    e1=1e-3
    revenue_loss=generate_final_revenue_loss(allocs, w, B, lamb, virtual_utility,two_index)
    mask = torch.zeros(n + 1, device=device)
    idx=0
    for i in range(n + 1):
        mask[i]=e1
        diff_w_revenue_loss = generate_final_revenue_loss(allocs, w+mask, B, lamb, virtual_utility,two_index)
        grad_w0[i]=(diff_w_revenue_loss-revenue_loss)/e1
        mask[i]=0
        idx+=1
        print(idx)
    mask = torch.zeros((n + 1, 2 ** m), device=device)
    for i in range(n + 1):
        for j in range(2**m):
            mask[i,j] = e1
            diff_b_revenue_loss = generate_final_revenue_loss(allocs, w, B, lamb + mask, virtual_utility,two_index)
            grad_b0[i,j] = (diff_b_revenue_loss - revenue_loss) / e1
            mask[i,j] = 0
            idx+=1
        print(idx)
    del mask
    return grad_w0,grad_b0
def RVVCAw(input_bids: torch.tensor, input_values: torch.tensor,device,h,l,theta,batchsize,w,lamb,max_ut):
    B, n, m = input_bids.shape
    utility = input_values - input_bids
    virtual_utility = torch.cat((utility, torch.zeros(B, 1, m).to(device)), axis=1).requires_grad_(True)  # B,N+1,m
    two_index = torch.tensor(np.fromfunction(lambda i: 2 ** i, (m,)), device=device)

    allocs = generate_all_deterministic_alloc(
        n + 1, m, device).unsqueeze(0).repeat(B, 1, 1, 1)


    num_epoch=5
    lr=0.1
    ut= generateceshi(n,m,32,device,"train",1000,w,lamb)
    print(ut)
    if (max_ut < ut):
        max_ut = ut
        torch.save(w, 'w.pt')
        torch.save(lamb, 'lamb.pt')

    for epoch in range(num_epoch):
        grad_w,grad_b=gradCalculate(allocs,w, B, lamb, virtual_utility,two_index,device)
        w=w-lr*grad_w
        lamb=lamb-lr*grad_b
        ut= generateceshi(n,m,32,device,"train",1000,w,lamb)
        print(ut)
        #l=generate_final_revenue_loss(allocs, w, B, lamb, virtual_utility)
        #print(l)
        if (max_ut < ut):
            #print(lamb)
            max_ut = ut
            torch.save(w, 'w.pt')
            torch.save(lamb, 'lamb.pt')
    return w,lamb,max_ut


torch.manual_seed(2024)
np.random.seed(2024)
n=int(2)
m=int(5)
bs=int(1024)
max_ut=0
device='cuda:0'
test_dir="test"
path_dir=os.path.join("data",str(n)+'x'+str(m))
path_dir = os.path.join(path_dir, test_dir)
test_data = load_data(path_dir)

test_bids = test_data[0]
test_values = test_data[1]
test_utility = torch.zeros(1).to(device)
test_payment = torch.zeros(1).to(device)
test_cost = torch.zeros(1).to(device)

l=(test_values.clone().detach()-test_bids.clone().detach()).min().cpu()
h=(test_values.clone().detach()-test_bids.clone().detach()).max().cpu()
theta=1
test_w=torch.zeros(n+1).to(device)
test_lamb=torch.zeros((n+1,2**m),device=device)

t = np.random.randint(0, math.ceil(math.log(h / l, (1 + theta))))
reserve_per_uti = l * ((1 + theta) ** t)
k = 0
per_virtual = np.zeros(2 ** m)
for i in range(m + 1):
    for j in range(math.comb(m, i)):
        per_virtual[k] = reserve_per_uti * i
        k = k + 1
lamb = torch.cat((torch.zeros((n, 2 ** m), device=device),
                  torch.tensor(per_virtual, device=device).to(torch.float32).unsqueeze(0)), 0).to(
    torch.float32)
w = torch.ones(n + 1, device=device).to(torch.float32)

for num in range(int(2)):
    print('----------------{}-----------------'.format(num))
    w,lamb,max_ut=RVVCAw(
        test_bids[num * bs:(num + 1) * bs].to(device),
        test_values[num * bs:(num + 1) * bs].to(device),device,h,l,theta,bs,w,lamb,max_ut)
