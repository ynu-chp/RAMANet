from docplex.mp.model import Model
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)


# 不可信opt
def opt_utility(bids,values):
    bidder, item = bids.shape[0], bids.shape[1]
    model = Model()
    # allocation = model.continuous_var_matrix(keys1=range(bidder),keys2=range(item), lb=0, ub=1, name='allocation')
    allocation = []
    for i in range(bidder):
        allocation.append(model.continuous_var_list([j for j in range(item)], lb=0,ub=1, name="flow_{}".format(i)))

    model.maximize(model.sum(allocation[i][j]*(values[i][j]-bids[i][j]) for i in range(bidder) for j in range(item)))  #
    for j in range(item):
        model.add_constraint(model.sum(allocation[i][j] for i in range(bidder))<=1.0)  #


    solution = model.solve()

    if (solution):
        return solution.objective_value
    else:
        return -1


def load_data(dir):
    data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
            np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]

    return tuple(data)


data_dir = "data"
scale = "2x5"

dir =  os.path.join(os.path.join(data_dir, scale),"final_test")  # get train data
data = load_data(dir)

bids = data[0]
values = data[1]

utility=0
for i in tqdm(range(bids.shape[0])):
    utility+=opt_utility(bids[i],values[i])
    # print(opt_utility(bids[i],values[i]))
utility/=bids.shape[0]
print(scale,"-opt-utility:",utility)
