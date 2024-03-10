import argparse
from collections import defaultdict
from joblib import Parallel, delayed
from dgl import save_graphs
import pickle
import dgl
import numpy as np


def generate_intent_forest(u, data_type):

    # 用户整个交互序列小于3，则不生成任何数据
    if len(interaction[u]) < 3:
        return

    # 负样本在dataloader的时候再进行采样
    if data_type == 'train':
        seq = np.zeros([args.L], dtype=int)
        pos = train[u][-1]
        idx = args.L - 1

        for item in reversed(train[u][:-1]):
            seq[idx] = item
            idx -= 1

            if idx < 0:
                break
    elif data_type == 'valid':
        seq = np.zeros([args.L], dtype=int)
        pos = valid[u]
        idx = args.L - 1

        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1

            if idx < 0:
                break
    elif data_type == 'test':
        seq = np.zeros([args.L], dtype=int)
        pos = test[u]
        idx = args.L - 1
        seq[idx] = valid[u]
        idx -= 1

        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1

            if idx < 0:
                break

    # 根据seq生成意图森林
    root = set()

    items, item_parents = [], []
    t_childs, t_parents = [], []

    idx = args.L - 1
    while idx >= 0 and seq[idx] != 0:
        pass

    # 保存数据
    if data_type == 'train':
        save_graphs()
    elif data_type == 'valid':
        save_graphs()
    elif data_type == 'test':
        save_graphs()


# 并行调用generate_intent_forest
def generate_train_data():
    user = interaction.keys()
    Parallel(n_jobs=args.job)(delayed(lambda u: generate_intent_forest(u, data_type='train'))(u) for u in user)


def generate_valid_data():
    user = interaction.keys()
    Parallel(n_jobs=args.job)(delayed(lambda u: generate_intent_forest(u, data_type='valid'))(u) for u in user)


def generate_test_data():
    user = interaction.keys()
    Parallel(n_jobs=args.job)(delayed(lambda u: generate_intent_forest(u, data_type='test'))(u) for u in user)


def data_partition(data_path):
    interaction = defaultdict(list)
    train, valid, test = {}, {}, {}

    with open(data_path, 'r') as f:
        for line in f.readlines():
            user, item = line.strip().split(' ')
            user, item = int(user), int(item)

            interaction[user].append(item)

    for user in interaction.keys():
        inter_num = len(interaction[user])
        if inter_num < 3:
            train[user] = interaction[user]
            valid[user] = []
            test[user] = []
        else:
            train[user] = interaction[user][:-2]
            valid[user] = [interaction[user][-2]]
            test[user] = [interaction[user][-1]]

    return interaction, train, valid, test


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='grocery')
    parse.add_argument('--job', type=int, default=10, help='parallel job num')
    parse.add_argument('--n_hop', type=int, default=2, help='category layer num')
    parse.add_argument('--L', type=int, default=50, help='max length of user interact sequence')

    args = parse.parse_args()

    data_path = f'dataset/{args.dataset}/{args.dataset}.txt'
    interaction, train, valid, test = data_partition(data_path)

    train_path = f'new_dataset/{args.dataset}_{args.n_hop}/train/'
    valid_path = f'new_dataset/{args.dataset}_{args.n_hop}/valid/'
    test_path = f'new_dataset/{args.dataset}_{args.n_hop}/test/'

    taxonomy_tree_path = f'dataset/{args.dataset}/item_taxonomy.dict'
    with open(taxonomy_tree_path, 'rb') as f:
        taxonomy_tree = pickle.load(f)

    generate_train_data()
    generate_valid_data()
    generate_test_data()
