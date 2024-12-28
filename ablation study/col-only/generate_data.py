import numpy as np
import torch
import os

def generate_data(sample_num, n_agents, m_items, path):
    value=np.random.rand(sample_num, n_agents, m_items)
    bid=np.random.normal(value,0.1*value)
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            for k in range(value.shape[2]):
                while bid[i,j,k]<0 or bid[i,j,k]>3*value[i,j,k]:
                    bid[i,j,k]=np.random.normal(value[i,j,k],0.1*value[i,j,k])
    value = value * 3

    path_dir=os.path.join("data",str(n_agents)+'x'+str(m_items))
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    path_dir = os.path.join(path_dir, path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    np.save(os.path.join(path_dir,"value"), value, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "bid"), bid, allow_pickle=True, fix_imports=True)


print("generate data:")

train_dir="train"
test_dir="test"
final_test_dir="final_test"

n=2
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))


n=4
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))

n=6
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))

n=8
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))

n=10
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))

n=12
m=5
train_sample_num=32768*2
test_sample_num=32768
final_sample_num=32768
generate_data(int(train_sample_num),int(n),int(m),train_dir)
generate_data(int(test_sample_num),int(n),int(m),test_dir)
generate_data(int(final_sample_num),int(n),int(m),final_test_dir)
print("bidder={},poi={},ok!".format(n,m))