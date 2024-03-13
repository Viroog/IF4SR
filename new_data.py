import argparse
from collections import defaultdict

import torch
from dgl import save_graphs
import pickle
import dgl
import numpy as np


def generate_intent_forest(u, data_type, first_taxonomy_num):
    # 用户整个交互序列小于4，则不生成任何数据
    # len(train) >= 2 & len(valid) == 1 & len(test) == 1 --> 最低要求
    if len(interaction[u]) < 4:
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
        pos = valid[u][0]
        idx = args.L - 1

        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1

            if idx < 0:
                break
    else:
        seq = np.zeros([args.L], dtype=int)
        pos = test[u][0]
        idx = args.L - 1
        seq[idx] = valid[u][0]
        idx -= 1

        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1

            if idx < 0:
                break

    # 根据seq生成意图森林
    # root的绝对id
    absolute_root = set()

    items, items_parent = [], []
    t_childs, t_parents = [], []

    idx = args.L - 1
    # 从后往前遍历，当遇到第一个为0的元素时说明该元素及其前面的元素都为0，不用为其生成森林了
    while idx >= 0 and seq[idx] != 0:
        item = seq[idx]
        item_taxonomies = taxonomy_tree[item]
        items.append(item)
        # 物品的父节点为最后一个标签，即上一级标签
        items_parent.append(item_taxonomies[-1])

        idx2 = len(item_taxonomies) - 1
        while idx2 > 0:
            t_childs.append(item_taxonomies[idx2])
            t_parents.append(item_taxonomies[idx2 - 1])
            idx2 -= 1

            # 根节点
            if idx2 == 0:
                absolute_root.add(item_taxonomies[idx2])

        idx -= 1

    forest_data = {('item', 'i2t', 'taxonomy'): (torch.LongTensor(items), torch.LongTensor(items_parent)),
                   ('taxonomy', 't2t', 'taxonomy'): (torch.LongTensor(t_childs), torch.LongTensor(t_parents))}

    forest = dgl.heterograph(forest_data)

    # 无重复的标签节点集合
    taxonomy_set = list(set(t_childs + t_parents))

    # 赋予树节点id属性
    max_item_id, max_taxonomy_id = max(items), max(taxonomy_set)
    # 右边界不包括
    forest.nodes['item'].data['id'] = torch.LongTensor(list(range(0, max_item_id + 1)))
    forest.nodes['taxonomy'].data['id'] = torch.LongTensor(list(range(0, max_taxonomy_id + 1)))

    # 提取子图，去除不必要的节点
    forest = dgl.node_subgraph(forest, {'item': list(set(items)), 'taxonomy': list(set(t_childs + t_parents))})

    # 相对root_id
    relative_root = torch.where(torch.isin(forest.nodes['taxonomy'].data['id'], torch.LongTensor(list(absolute_root))))[
        0]

    # 剩余用-1进行填充
    pad_relative_root = torch.full((first_taxonomy_num,), -1)
    pad_relative_root[:relative_root.shape[0]] = relative_root

    # 保存数据 直接保存tensor
    if data_type == 'train':
        save_graphs(f'./new_dataset/{args.dataset}/train/{u}.bin', forest,
                    {'user': torch.LongTensor([u]), 'seq': torch.LongTensor(seq), 'target': torch.LongTensor([pos]),
                     'root': torch.LongTensor(pad_relative_root)})
    elif data_type == 'valid':
        save_graphs(f'./new_dataset/{args.dataset}/valid/{u}.bin', forest,
                    {'user': torch.LongTensor([u]), 'seq': torch.LongTensor(seq), 'target': torch.LongTensor([pos]),
                     'root': torch.LongTensor(pad_relative_root)})
    elif data_type == 'test':
        save_graphs(f'./new_dataset/{args.dataset}/test/{u}.bin', forest,
                    {'user': torch.LongTensor([u]), 'seq': torch.LongTensor(seq), 'target': torch.LongTensor([pos]),
                     'root': torch.LongTensor(pad_relative_root)})


# 并行调用generate_intent_forest
# def generate_train_data():
#     user = interaction.keys()
#     Parallel(n_jobs=args.job)(delayed(lambda u: generate_intent_forest(u, data_type='train'))(u) for u in user)
#
#
# def generate_valid_data():
#     user = interaction.keys()
#     Parallel(n_jobs=args.job)(delayed(lambda u: generate_intent_forest(u, data_type='valid'))(u) for u in user)
#
#
# def generate_test_data():
#     user = interaction.keys()
#     Parallel(n_jobs=args.job)(delayed(lambda u: generate_intent_forest(u, data_type='test'))(u) for u in user)


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

    first_taxonomy_set = set()

    for item, taxonomies in taxonomy_tree.items():
        for idx, taxonomy in enumerate(taxonomies):
            if idx == 0:
                first_taxonomy_set.add(taxonomy)

    user = interaction.keys()
    for u in user:
        generate_intent_forest(u, data_type='train', first_taxonomy_num=len(first_taxonomy_set))
        generate_intent_forest(u, data_type='valid', first_taxonomy_num=len(first_taxonomy_set))
        generate_intent_forest(u, data_type='test', first_taxonomy_num=len(first_taxonomy_set))
