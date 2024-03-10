import copy
import random

import dgl
import torch
from collections import defaultdict
from multiprocessing import Process, Queue
import numpy as np


# 生成单个用户的意图森林
# 先别管什么训练集、验证集和测试集，先将整个序列的意图树构建出来
def generate_user_intent_forest(user, seq, taxonomy_tree):
    # 森林中每棵树的根节点集合
    root = set()

    items, items_parent = [], []
    t_childs, t_parents = [], []

    # 构建成异构图，物品到标签是i2t（item_to_taxonomy），子标签到父标签是t2t（taxonomy_to_taxnomy）
    for item in seq:
        # [first_category, second_category, ...]
        item_taxonomies = taxonomy_tree[item]
        items.append(item)
        items_parent.append(item_taxonomies[-1])

        i = len(item_taxonomies) - 1
        while i > 0:
            t_childs.append(item_taxonomies[i])
            t_parents.append(item_taxonomies[i - 1])
            i -= 1

            # 加入根节点
            if i == 0:
                root.add(item_taxonomies[i])

    forest_data = {('item', 'i2t', 'taxonomy'): (torch.tensor(items), torch.tensor(items_parent)),
                   ('taxonomy', 't2t', 'taxonomy'): (torch.tensor(t_childs), torch.tensor(t_parents))}
    # 这样建出来的森林有一个问题是多了很多零散的节点，这些节点出度入度均为0
    # 根节点：出度为0，入度不为0        叶子节点：出度不为0，入度为0     其余节点：出度入度均不为0
    forest = dgl.heterograph(forest_data)

    # 无重复的叶子节点集合以及无重复的标签节点集合
    item_set, taxonomy_set = list(set(items)), list(set(t_childs + t_parents))

    # 赋予树节点id属性
    max_item_id, max_taxonomy_id = max(items), max(taxonomy_set)
    # 右边界不包括
    forest.nodes['item'].data['id'] = torch.LongTensor(list(range(0, max_item_id + 1)))
    forest.nodes['taxonomy'].data['id'] = torch.LongTensor(list(range(0, max_taxonomy_id + 1)))

    # 提取子图，去除不必要的节点
    forest = dgl.node_subgraph(forest, {'item': list(set(items)), 'taxonomy': list(set(t_childs + t_parents))})

    return list(root), forest


# 参照SASRec源码
def sample_function(train, taxonomy_tree, user_num, item_num, batch_size, L, result_queue, SEED):
    def sample():
        # [left, right)
        user = np.random.randint(1, user_num + 1)
        while len(train[user]) <= 1:
            user = np.random.randint(1, user_num + 1)

        seq = np.zeros([L], dtype=np.int32)
        pos = train[user][-1]
        neg = np.random.randint(1, item_num + 1)
        while neg in train[user]:
            neg = np.random.randint(1, item_num + 1)
        idx = L - 1

        for item in reversed(train[user][:-1]):
            seq[idx] = item
            idx -= 1

            if idx < 0:
                break

        root, forest = generate_user_intent_forest(user, train[user][:-1], taxonomy_tree)

        return user, seq, pos, neg, root, forest

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


# 参照SASRec源码
class Sampler(object):
    def __init__(self, train, taxonomy_tree, user_num, item_num, batch_size, L, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(train, taxonomy_tree, user_num, item_num, batch_size, L, self.result_queue,
                              np.random.randint(2e9)))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate_valid(model, dataset, taxonomy_tree, args):
    train, valid, test, user_num, item_num = copy.deepcopy(dataset)

    ndcg = 0.0
    hit = 0.0
    valid_user = 0.0

    if user_num > 10000:
        users = random.sample(range(1, user_num + 1), 10000)
    else:
        users = range(1, user_num + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.L], dtype=int)
        idx = args.L - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, item_num + 1)
            while t in rated:
                t = np.random.randint(1, item_num + 1)
                item_idx.append(t)

        # 需要建树的物品为seq中不为0的元素
        root, forest = generate_user_intent_forest(u, list(seq[seq != 0]), taxonomy_tree)

        predictions = -model.predict()
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            hit += 1

    return ndcg / valid_user, hit / valid_user


def evaluate(model, dataset, tree_taxonomy, args):
    train, valid, test, user_num, item_num = copy.deepcopy(dataset)

    ndcg = 0.0
    hit = 0.0
    valid_user = 0.0

    if user_num > 10000:
        users = random.sample(range(1, user_num + 1), 10000)
    else:
        users = range(1, user_num + 1)

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.L], dtype=int)
        idx = args.L - 1
        seq[idx] = valid[u][0]
        idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, item_num + 1)
            while t in rated:
                t = np.random.randint(1, item_num + 1)
            item_idx.append(t)

        root, forest = generate_user_intent_forest(u, list(seq[seq != 0]), tree_taxonomy)

        predictions = -model.predict()
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            hit += 1

    return ndcg / valid_user, hit / valid_user


# 参照SASRec源码
def data_partition(dataset):
    user_num, item_num = 0, 0

    data_path = f'./dataset/{dataset}/{dataset}.txt'
    interaction = defaultdict(list)
    train, valid, test = {}, {}, {}

    with open(data_path, 'r') as f:
        for line in f.readlines():
            # user, item都是从1开始计数且连续，因此id大小就是数量
            user, item = line.strip().split(' ')
            user, item = int(user), int(item)
            user_num = max(user_num, user)
            item_num = max(item_num, item)

            interaction[user].append(item)

    for user in interaction.keys():
        interaction_num = len(interaction[user])
        if interaction_num < 3:
            train[user] = interaction[user]
            valid[user] = []
            test[user] = []
        else:
            train[user] = interaction[user][:-2]
            valid[user] = [interaction[user][-2]]
            test[user] = [interaction[user][-1]]

    return train, valid, test, user_num, item_num
