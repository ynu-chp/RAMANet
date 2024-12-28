import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from net import TransformerMechanism



class RAMANet(nn.Module):
    def __init__(self, args, oos=False) -> None:
        super().__init__()
        self.n_agents = args.n_agents
        self.m_items = args.m_items

        self.device = args.device
        self.menu_size = args.menu_size
        self.bool_RVCGNet=args.bool_RVCGNet

        self.alloc_softmax_temperature = args.alloc_softmax_temperature

        mask = 1 - torch.eye((self.n_agents)).to(self.device)#(n,n)
        self.mask = torch.zeros(args.n_agents, args.batch_size, args.n_agents).to(self.device)#(n, bs, n)
        for i in range(args.n_agents):
            self.mask[i] = mask[i].repeat(args.batch_size, 1)


        self.mask = self.mask.reshape(args.n_agents * args.batch_size, args.n_agents)#(n * bs, n)


        self.mechanism = TransformerMechanism(args.n_layer, args.n_head, args.d_hidden,
                args.menu_size).to(self.device)


    def test_time_forward(self, input_bids: torch.tensor, input_values: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        input_bids: B, n, m  (bs,n,m)
        X: B, n_agents, dx  (bs,n,dx)
        Y: B, m_items, dy   (bs,m,dy)
        '''
        B, n, m = input_bids.shape
        value_bid=input_values-input_bids #(bs,n,m)

        allocs, w, b = self.mechanism(input_bids,input_values, self.alloc_softmax_temperature)
        # (bs,ms,n,m)   (bs,n)  (bs,ms)
        if self.bool_RVCGNet == True:
            w = torch.ones(w.shape).to(self.device)
            b = torch.zeros(b.shape).to(self.device)

        allocs = torch.cat((allocs, torch.zeros(B, 1, n, m).to(self.device)), 1) # bs, ms, n, m
        b = torch.cat((b, torch.zeros((B, 1)).to(self.device)), 1) # bs,ms
        assert w.all() > 0

        util_from_items = (allocs * value_bid.unsqueeze(1)).sum(axis=-1) # bs,ms,n
        per_agent_welfare = w.unsqueeze(1) * util_from_items #  bs,ms,n
        total_welfare = per_agent_welfare.sum(axis=-1) # bs,ms
        alloc_choice_ind = torch.argmax(total_welfare + b, -1)  #bs

        item_allocation = [allocs[i, alloc_choice_ind[i],...] for i in range(B)]
        item_allocation = torch.stack(item_allocation) # bs,n,m

        chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(B)]
        chosen_alloc_welfare_per_agent = torch.stack(chosen_alloc_welfare_per_agent) # bs,n

        removed_alloc_choice_ind_list = []

        #payment
        payments = []
        for i in range(self.n_agents):
            mask = torch.ones(n).to(self.device)#(n,)
            mask[i] = 0

            removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, n)#(bs,ms,n)
            total_removed_welfare = removed_i_welfare.sum(-1) #(bs,ms)
            removed_alloc_choice_ind = torch.argmax(total_removed_welfare + b, -1) #(bs,)
            removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in range(B)]#
            removed_chosen_welfare = torch.stack(removed_chosen_welfare)# (bs,)
                
            removed_alloc_b = [b[i, removed_alloc_choice_ind[i]] for i in range(B)]
            removed_alloc_b = torch.stack(removed_alloc_b) #(bs,)

            alloc_b = [b[i, alloc_choice_ind[i]] for i in range(B)]
            alloc_b = torch.stack(alloc_b)#bs

            payments.append(
                (1.0 / w[:,i])
                * (
                    (chosen_alloc_welfare_per_agent.sum(1) + alloc_b)
                    -( removed_chosen_welfare + removed_alloc_b )
                )

            )
            removed_alloc_choice_ind_list.append(removed_alloc_choice_ind)

        payments = torch.stack(payments)# (n,bs)

        payments = payments + (input_bids * item_allocation).sum(-1).permute(1,0)
        utility=(input_values*item_allocation).sum(-1).permute(1,0)-payments
        return alloc_choice_ind, item_allocation, utility, payments, allocs, w, b, removed_alloc_choice_ind_list, (input_bids * item_allocation).sum(-1).permute(1,0)

    
    def forward(self, input_bids: torch.tensor, input_values: torch.tensor, softmax_temp: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''                      parser.add_argument('--init_softmax_temperature', type=int, default=500)
        input_bids: B, n, m 
        X: B, n_agents, dx 
        Y: B, m_items, dy
        '''
        B, n, m = input_bids.shape
        value_bid = input_values - input_bids  # (bs,n,m)


        allocs, w, b = self.mechanism(input_bids,input_values, self.alloc_softmax_temperature)

        if self.bool_RVCGNet == True:
            w = torch.ones(w.shape).to(self.device)
            b = torch.zeros(b.shape).to(self.device)


        allocs = torch.cat((allocs, torch.zeros(B, 1, n, m).to(self.device)), 1) # bs, ms, n, m
        b = torch.cat((b, torch.zeros((B, 1)).to(self.device)), 1) # bs,ms


        util_from_items = (allocs * value_bid.unsqueeze(1)).sum(axis=-1) # bs,ms,n
        per_agent_welfare = w.unsqueeze(1) * util_from_items #(bs,ms,n)
        total_welfare = per_agent_welfare.sum(axis=-1) #(bs,ms)

        alloc_choice = F.softmax((total_welfare + b) * softmax_temp, dim=-1) # bs,ms

        item_allocation = (torch.unsqueeze(torch.unsqueeze(alloc_choice, -1), -1) * allocs).sum(axis=1)# (bs,n,m)

        chosen_alloc_welfare_per_agent = (per_agent_welfare * torch.unsqueeze(alloc_choice, -1)).sum(axis=1) #(bs,n)

        n_chosen_alloc_welfare_per_agent= chosen_alloc_welfare_per_agent.repeat(n, 1)#(n*bs, n)

        masked_chosen_alloc_welfare_per_agent = n_chosen_alloc_welfare_per_agent * self.mask #  nB*bs, n

        n_per_agent_welfare = per_agent_welfare.repeat(n, 1, 1)#(n*bs, ms, n)

        removed_i_welfare = n_per_agent_welfare * self.mask.reshape(n*B, 1, n) #(n*bs, ms, n)

        total_removed_welfare  = removed_i_welfare.sum(axis=-1)  # (n*bs,ms)
                                #
        removed_alloc_choice = F.softmax((total_removed_welfare + b.repeat(n, 1)) * softmax_temp, dim=-1)# (n*bs,ms)

        removed_chosen_welfare_per_agent = (

            removed_i_welfare * removed_alloc_choice.unsqueeze(-1)
        ).sum(axis=1)# (n*bs,n)

        payments = torch.zeros(n * B).to(self.device)

        payments = (1 / w.permute(1, 0).reshape(n * B)) * (
            ( n_chosen_alloc_welfare_per_agent.sum(-1)
            +(alloc_choice * b).sum(1).repeat(n))
            -
            (removed_chosen_welfare_per_agent.sum(-1)

            + (removed_alloc_choice * b.repeat(n, 1)).sum(-1))
        ) # n*bs
        payments = payments.reshape(n, B)+(item_allocation*input_bids).sum(-1).permute(1,0)
        utility = (input_values * item_allocation).sum(-1).permute(1, 0) - payments
        return alloc_choice, item_allocation, utility, payments, allocs,(item_allocation*input_bids).sum(-1).permute(1,0)


