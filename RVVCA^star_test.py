import math

import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import logging
import math

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
    return torch.tensor(alloc).to(torch.float32).to(device)

def generate_result(allocs,input_bids,input_values,w,lamb,B,device,n,m):
    virtual_utility = torch.cat((input_values - input_bids, torch.zeros(B, 1, m).to(device)), axis=1)


    temp_w = w.reshape(1, 1, n + 1).repeat(B, (n + 1) ** m, 1)

    two_index = torch.tensor(np.fromfunction(lambda i: 2 ** i, (m,)), device=device)

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
        removed_util_from_items = ((allocs* mask.reshape(1, 1, n+1,m) * virtual_utility.unsqueeze(1)).sum(axis=-1)) * temp_w + allocs_two_value

        total_removed_welfare = removed_util_from_items.sum(-1)
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
        )
        mask[i]=one_mask

    payments = torch.stack(payments)

    payments = payments + (input_bids * item_allocation[:,:-1,:]).sum(-1).permute(1, 0)
    utility = (input_values * item_allocation[:,:-1,:]).sum(-1).permute(1, 0) - payments

    del two_index
    del allocs_two_index
    del allocs_two_value

    return utility.sum(0).mean(), payments.sum(0).mean(), (input_bids * item_allocation[:,:-1,:]).sum(-1).permute(1,0).sum(0).mean()

def generateceshi(n,m,bs,device,test_dir,total_times,w,lamb):
    path_dir=os.path.join("data",str(n)+'x'+str(m))

    path_dir = os.path.join(path_dir, test_dir)
    test_data = load_data(path_dir).to(device)


    test_bids = test_data[0]
    test_values = test_data[1]
    test_utility = torch.zeros(1).to(device)
    test_payment = torch.zeros(1).to(device)
    test_cost = torch.zeros(1).to(device)

    allocs = generate_all_deterministic_alloc(n+1,m,device,unit_demand = True).unsqueeze(0).repeat(bs, 1, 1, 1)


    for num in range(int(total_times/ bs)):
        utility,payment,cost=generate_result(allocs,test_bids[num * bs:(num + 1) * bs],test_values[num * bs:(num + 1) * bs],w,lamb,bs,device,n,m)
        #print("utility:",utility,"payment:",payment,"cost:",cost)
        test_utility += utility
        test_payment += payment
        test_cost += cost
    test_utility /= int(total_times/bs)
    test_payment /= int(total_times/bs)
    test_cost /= int(total_times/bs)
    print(f"RVVCA:n_bidder={n},m_poi={m}, test_utility: {test_utility}," f"test_payment: {test_payment}," f"test_cost: {test_cost}")
    return test_utility

if __name__ == "__main__":
    device='cuda:0'
    n=4
    m=5
    path_pt_dir = os.path.join("pt", str(n) + 'X' + str(m))
    w_pt_dir = os.path.join(path_pt_dir, "w.pt")
    lamb_pt_dir = os.path.join(path_pt_dir, "lamb.pt")
    w = torch.load(w_pt_dir).to(device)
    lamb = torch.load(lamb_pt_dir).to(device)
    generateceshi(n,m,128,device,"final_test",10000,w,lamb)