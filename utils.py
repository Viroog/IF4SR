import copy
import os
import random

import dgl
import torch
from collections import defaultdict
from multiprocessing import Process, Queue
import numpy as np
from torch.utils.data import Dataset


# 生成单个用户的意图森林
# 先别管什么训练集、验证集和测试集，先将整个序列的意图树构建出来
def generate_user_intent_forest(user, seq, taxonomy_tree, first_taxonomy_num):
    # 森林中每棵树的根节点集合
    absolute_root = set()

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
                absolute_root.add(item_taxonomies[i])

    forest_data = {('item', 'i2t', 'taxonomy'): (torch.LongTensor(items), torch.LongTensor(items_parent)),
                   ('taxonomy', 't2t', 'taxonomy'): (torch.LongTensor(t_childs), torch.LongTensor(t_parents))}
    # 这样建出来的森林有一个问题是多了很多零散的节点，这些节点出度入度均为0
    # 根节点：出度为0，入度不为0        叶子节点：出度不为0，入度为0     其余节点：出度入度均不为0
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

    # 绝对root_id
    # return list(absolute_root), forest
    # 真实root_id在图中对应的节点id，
    # tmp = forest.nodes['taxonomy'].data['id']
    # tmp2 = torch.LongTensor(list(root))
    # res = torch.where(torch.isin(tmp, tmp2))

    # 相对root_id
    relative_root = torch.where(torch.isin(forest.nodes['taxonomy'].data['id'], torch.LongTensor(list(absolute_root))))[0]

    pad_relative_root = torch.full((first_taxonomy_num,), -1)
    pad_relative_root[:relative_root.shape[0]] = relative_root

    return pad_relative_root.detach().numpy(), forest


# 参照SASRec源码
def sample_function(train, taxonomy_tree, user_num, item_num, batch_size, L, first_taxonomy_num, result_queue, SEED):
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

        root, forest = generate_user_intent_forest(user, train[user][:-1], taxonomy_tree, first_taxonomy_num)

        return user, seq, pos, neg, root, forest

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


# 参照SASRec源码
class Sampler(object):
    def __init__(self, train, taxonomy_tree, user_num, item_num, batch_size, L, first_taxonomy_num, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(
                            train, taxonomy_tree, user_num, item_num, batch_size, L, first_taxonomy_num,
                            self.result_queue,
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

        seq = np.zeros([args.L], dtype=np.int32)
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

        # 全部操作都需要增加一个维度
        seq = np.expand_dims(np.array(seq), axis=0)
        item_idx = np.expand_dims(np.array(np.array(item_idx)), axis=0)
        root = [root]
        batch_forest = dgl.batch([forest])

        predictions = -model.predict(seq, item_idx, root, batch_forest)
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

        seq = np.zeros([args.L], dtype=np.int32)
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

        seq = np.expand_dims(np.array(seq), axis=0)
        item_idx = np.expand_dims(np.array(np.array(item_idx)), axis=0)
        root = [root]
        batch_forest = dgl.batch([forest])

        predictions = -model.predict(seq, item_idx, root, batch_forest)
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


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()

    for file in dir_list:
        data_dir.append(os.path.join(data_path, file))

    return data_dir


def generate_neg(l, r, interact):
    neg = np.random.randint(l, r)
    while neg in interact:
        neg = np.random.randint(l, r)

    return neg


# new_main的数据加载方式
class myFloder(Dataset):
    def __init__(self, root_dir, loader, interact, item_num):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.interact = interact
        self.item_num = item_num
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data, self.item_num, self.interact

    def __len__(self):
        return self.size


def collate(data):
    forest, user, seq, pos, neg, root = [], [], [], [], [], []

    for d in data:
        forest.append(d[0][0][0])
        user.append(d[0][1]['user'])
        # 加上detach()表明不需要计算梯度, 可能服务器和自己电脑上的numpy版本不一样
        seq.append(d[0][1]['seq'].detach().numpy())
        pos.append(d[0][1]['target'])
        neg.append(generate_neg(1, d[1] + 1, d[2][user[-1].item()]))
        # root是个张量，每个元素是一个张量
        root.append(d[0][1]['root'].detach().numpy())

    # 在collate中完成所有数据格式的转化，会大大减少IO时间，能够更加有效利用GPU
    # 否则I/O时间是模型训练时间的十倍甚至百倍
    return dgl.batch(forest), torch.LongTensor(user), torch.LongTensor(np.array(seq)), torch.LongTensor(
        pos), torch.LongTensor(neg), torch.LongTensor(np.array(root))


def collate_valid_test(data):
    forest, user, seq, pos, neg, root = [], [], [], [], [], []

    for d in data:
        forest.append(d[0][0][0])
        user.append(d[0][1]['user'])
        seq.append(d[0][1]['seq'].detach().numpy())
        root.append(d[0][1]['root'].detach().numpy())

    return dgl.batch(forest), torch.LongTensor(user), torch.LongTensor(np.array(seq)), torch.LongTensor(), torch.LongTensor(np.array(root))