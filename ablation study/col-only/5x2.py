import argparse
import torch
from auction import RAMANet
from tqdm import tqdm
import logging
import os
import numpy as np
from logger import get_logger


# logging.basicConfig(
# level=logging.INFO,
# format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
# datefmt="%Y-%m-%d,%H:%M:%S",
# )

def str2bool(v):
    return v.lower() in ('true', '1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/5x2')
    parser.add_argument('--training_set', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')
    parser.add_argument('--final_test_set', type=str, default='final_test')

    parser.add_argument('--n_agents', type=int, default=5)
    parser.add_argument('--m_items', type=int, default=2)
    parser.add_argument('--menu_size', type=int, default=128)
    parser.add_argument('--bool_RVCGNet', type=str2bool, default=False)

    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=32)
    parser.add_argument('--init_softmax_temperature', type=int, default=500)
    parser.add_argument('--alloc_softmax_temperature', type=int, default=1)

    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_steps', type=int, default=4000)
    parser.add_argument('--train_sample_num', type=int, default=32768 * 2)
    parser.add_argument('--eval_freq', type=int, default=50)  # 500
    parser.add_argument('--eval_sample_num', type=int, default=32768)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--decay_round_one', type=int, default=2000)  #
    parser.add_argument('--one_lr', type=float, default=5e-5)  #
    parser.add_argument('--decay_round_two', type=int, default=3000)  #
    parser.add_argument('--two_lr', type=float, default=1e-5)  #
    parser.add_argument('--bool_test', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='./results')
    parser.add_argument('--final_test_batch_size', type=int, default=2048)

    return parser.parse_args()


def load_data(dir):
    data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
            np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]

    return tuple(data)


if __name__ == "__main__":
    args = parse_args()

    file_path = f"{args.n_agents}_{args.m_items}_{args.menu_size}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    log_path = f"{file_path}/record.log"
    logger = get_logger(log_path)
    logger.info(args)

    torch.manual_seed(args.seed)
    DEVICE = args.device

    train_dir = os.path.join(args.data_dir, args.training_set)  #get train data
    train_data = load_data(train_dir)

    test_dir = os.path.join(args.data_dir, args.test_set)  # get test data
    test_data = load_data(test_dir)

    final_test_dir = os.path.join(args.data_dir, args.final_test_set)  # get final test data
    final_test_data = load_data(final_test_dir)

    model = RAMANet(args).to(DEVICE)
    # if args.bool_test:
    #     state_dict = torch.load('model/*x*',map_location='cuda:0')
    #     model.mechanism.load_state_dict(state_dict)

    cur_softmax_temperature = args.init_softmax_temperature
    warm_up_init = 1e-8
    warm_up_end = args.lr
    warm_up_anneal_increase = (warm_up_end - warm_up_init) / 100
    optimizer = torch.optim.Adam(model.mechanism.parameters(), lr=warm_up_init)

    bs = args.batch_size
    num_per_train = int(
        args.train_sample_num / bs)
    for i in tqdm(range(args.train_steps)):
        if i == args.train_steps - 1 or (i >= 1000 and (
                i % args.eval_freq == 0)):
            if i == args.train_steps - 1:
                if not os.path.exists("model"):
                    os.makedirs("model")


                model_path =os.path.join("model",
                                                       str(args.n_agents) + "x"+str(args.m_items))
                torch.save(model.mechanism.state_dict(), model_path)
            with torch.no_grad():

                test_bids = test_data[0]
                test_values = test_data[1]
                test_utility = torch.zeros(1).to(DEVICE)
                test_payment = torch.zeros(1).to(DEVICE)
                test_cost = torch.zeros(1).to(DEVICE)
                for num in range(int(test_values.shape[0] / bs)):
                    choice_id, _, utility, payment, allocs, _, _, _, cost = model.test_time_forward(
                        torch.tensor(test_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                        torch.tensor(test_values[num * bs:(num + 1) * bs]).to(DEVICE))
                    test_utility += utility.sum()
                    test_payment += payment.sum()
                    test_cost += cost.sum()
                test_utility /= test_values.shape[0]
                test_payment /= test_values.shape[0]
                test_cost /= test_values.shape[0]
                logger.info(
                    f"step {i}: test_utility: {test_utility}," f"test_payment: {test_payment}," f"test_cost: {test_cost}")


        train_bids = train_data[0]
        train_values = train_data[1]

        reportloss = 0
        train_utility = 0
        train_payment = 0
        train_cost = 0
        for num in range(num_per_train):
            optimizer.zero_grad()
            _, _, utility, payment, allocs, cost = model(torch.tensor(train_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                                                         torch.tensor(train_values[num * bs:(num + 1) * bs]).to(DEVICE),
                                                         cur_softmax_temperature)
            loss = - utility.sum(0).mean()
            reportloss += loss.data
            train_utility += utility.sum(0).mean().data
            train_payment += payment.sum(0).mean().data
            train_cost += cost.sum(0).mean().data
            loss.backward()
            optimizer.step()

        if i % 1 == 0:
            logger.info(f"step {i}: loss: {reportloss / num_per_train},"
                        f"train_utility: {train_utility / num_per_train},"
                        f"train_payment: {train_payment / num_per_train}," f"train_cost: {train_cost / num_per_train}")

        if i <= 100:  # warm up
            for p in optimizer.param_groups:
                p['lr'] += warm_up_anneal_increase  # 预热

        if i == args.decay_round_one:
            for p in optimizer.param_groups:
                p['lr'] = args.one_lr

        if i == args.decay_round_two:
            for p in optimizer.param_groups:
                p['lr'] = args.two_lr

    # test
    logger.info("------------Final test------------")
    with torch.no_grad():

        final_test_bids = final_test_data[0]
        final_test_values = final_test_data[1]
        final_test_utility = torch.zeros(1).to(DEVICE)
        final_test_payment = torch.zeros(1).to(DEVICE)
        final_test_cost = torch.zeros(1).to(DEVICE)
        for num in range(int(final_test_values.shape[0] / bs)):
            choice_id, _, utility, payment, allocs, _, _, _, cost = model.test_time_forward(
                torch.tensor(final_test_bids[num * bs:(num + 1) * bs]).to(DEVICE),
                torch.tensor(final_test_values[num * bs:(num + 1) * bs]).to(DEVICE))
            final_test_utility += utility.sum()
            final_test_payment += payment.sum()
            final_test_cost += cost.sum()
        final_test_utility /= final_test_values.shape[0]
        final_test_payment /= final_test_values.shape[0]
        final_test_cost /= final_test_values.shape[0]
        logger.info(
            f"final_test_utility: {final_test_utility}," f"final_test_payment: {final_test_payment}," f"final_test_cost: {final_test_cost}")

