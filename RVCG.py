
import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import logging

def generate_all_deterministic_alloc(n_agents, m_items,unit_demand = False) -> torch.tensor: # n buyers, m items -> alloc (n+1, m)
    alloc_num = (n_agents+1) ** (m_items)
    def gen(t, i, j):
        x = (n_agents+1) ** (m_items - 1 - j)
        return np.where((t // x) % (n_agents+1) == i, 1.0, 0.0)
    alloc = np.fromfunction(gen, (alloc_num-1, n_agents, m_items))
    return torch.tensor(alloc).to(torch.float32)

def load_data(dir):
    data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
            np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]
    return torch.tensor(np.array(data)).to(torch.float32)

def test_time_forward( input_bids: torch.tensor, input_values: torch.tensor):
    """
    input_bids: B, n, m  (bs,n,m)
    X: B, n_agents, dx  (bs,n,dx)
    Y: B, m_items, dy   (bs,m,dy)
    """
    B, n, m = input_bids.shape
    value_bid =input_values -input_bids








    allocs=generate_all_deterministic_alloc(n,m).unsqueeze(0).repeat(B,1,1,1)


    allocs = torch.cat((allocs, torch.zeros(B, 1, n, m)), 1)



    util_from_items = (allocs * value_bid.unsqueeze(1)).sum(axis=-1)

    per_agent_welfare =  util_from_items
    total_welfare = per_agent_welfare.sum(axis=-1)
    alloc_choice_ind = torch.argmax(total_welfare , -1)

    item_allocation = [allocs[i, alloc_choice_ind[i] ,...] for i in range(B)]
    item_allocation = torch.stack(item_allocation)

    chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(B)]
    chosen_alloc_welfare_per_agent = torch.stack \
        (chosen_alloc_welfare_per_agent)


    removed_alloc_choice_ind_list = []


    payments = []
    for i in range(n)  :
        mask = torch.ones(n)
        mask[i] = 0

        removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, n  )
        total_removed_welfare = removed_i_welfare.sum(-1)
        removed_alloc_choice_ind = torch.argmax(total_removed_welfare , -1)
        removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in range(B)  ]
        removed_chosen_welfare = torch.stack(removed_chosen_welfare  )







        payments.append(
             (
                    (chosen_alloc_welfare_per_agent.sum(1)) -( removed_chosen_welfare )

            )

        )
        removed_alloc_choice_ind_list.append(removed_alloc_choice_ind)

    payments = torch.stack(payments)

    payments = payments + (input_bids * item_allocation).sum(-1).permute(1,0)
    utility=(input_values*item_allocation).sum (-1).permute(1,0)-payments
    return utility, payments, (input_bids * item_allocation).sum(-1).permute(1,0)


n=int(4)
m=int(5)
bs=int(128)
test_dir="test"
path_dir=os.path.join("data",str(n)+'x'+str(m))
path_dir = os.path.join(path_dir, test_dir)
test_data = load_data(path_dir)

test_bids = test_data[0]
test_values = test_data[1]
test_utility = torch.zeros(1)
test_payment = torch.zeros(1)
test_cost = torch.zeros(1)
for num in range(int(test_values.shape[0] / bs)):
    utility, payment,  cost = test_time_forward(
        test_bids[num * bs:(num + 1) * bs].clone().detach() ,
        test_values[num * bs:(num + 1) * bs].clone().detach())
    test_utility += utility.sum()
    test_payment += payment.sum()
    test_cost += cost.sum()
test_utility /= test_values.shape[0]
test_payment /= test_values.shape[0]
test_cost /= test_values.shape[0]
print(f"RVCG:n_bidder={n},m_poi={m}, test_utility: {test_utility}," f"test_payment: {test_payment}," f"test_cost: {test_cost}")