import pickle
import dgl
import torch
from collections import defaultdict
from multiprocessing import Process, Queue
import numpy as np

import pandas as pd


# 生成单个用户的意图森林
# 先别管什么训练集、验证集和测试集，先将整个序列的意图树构建出来
def generate_user_intent_forest(user, seq):
    # 森林中每棵树的根节点集合以及叶子节点集合
    root, leaf = set(), set()

    items, items_parent = [], []
    t_childs, t_parents = [], []

    # 构建成异构图，物品到标签是i2t（item_to_taxonomy），子标签到父标签是t2t（taxonomy_to_taxnomy）
    for item in seq:
        leaf.add(item)

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

    return root, leaf, forest


# 生成意图森林
def generate_intent_forest(data):
    users = data['user'].unique()

    for user in users:
        generate_user_intent_forest(user, data.loc[data['user'] == user]['item'].values)

    return None


def random_neg(l, r, s):
    pass


# 参照SASRec源码
def sample_function(train, valid, user_num, item_num, batch_size, maxlen, result_queue, SEED):
    def sample():
        # [left, right)
        user = np.random.randint(1, user_num + 1)
        while len(train[user] <= 1):
            user = np.random.randint(1, user_num + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        nxt = train[user][-1]
        idx = maxlen - 1

        root, leaf, forest = generate_user_intent_forest(user, train[user])


    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


# 参照SASRec源码
class Sampler(object):
    def __init__(self, train, valid, user_num, item_num, batch_size, maxlen, n_workes=1):
        self.result_queue = Queue(maxsize=n_workes * 10)
        self.processors = []
        for i in range(n_workes):
            self.processors.append(
                Process(target=sample_function,
                        args=(train, valid, user_num, item_num, batch_size, maxlen, self.result_queue, np.random.randint(2e9)))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


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
