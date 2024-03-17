import os
import pickle

import dgl
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()

    for dir in dir_list:
        for file in os.listdir(os.path.join(data_path, dir)):
            data_dir.append(os.path.join(os.path.join(data_path, dir), file))

    return data_dir


# new_main的数据加载方式
class myFloder(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size


def collate(data):
    forest, user, seq, target, root = [], [], [], [], []

    for d in data:
        forest.append(d[0][0])
        user.append(d[1]['user'])
        # 加上detach()表明不需要计算梯度, 可能服务器和自己电脑上的numpy版本不一样
        # seq.append(d[0][1]['seq'].detach().numpy())
        seq.append(d[1]['seq'].detach().numpy())
        target.append(d[1]['target'])
        # root是个张量，每个元素是一个张量
        root.append(d[1]['root'].detach().numpy())

    # 在collate中完成所有数据格式的转化，会大大减少IO时间，能够更加有效利用GPU
    # 否则I/O时间是模型训练时间的十倍甚至百倍
    return dgl.batch(forest), torch.LongTensor(user), torch.LongTensor(np.array(seq)), torch.LongTensor(
        target), torch.LongTensor(
        np.array(root))


def collate_valid_test(data, user_dict, item_set):
    forest, user, seq, target, root, neg_items = [], [], [], [], [], []

    for d in data:
        forest.append(d[0][0])
        user.append(d[1]['user'])
        seq.append(d[1]['seq'].detach().numpy())
        target.append(d[1]['target'])
        neg_items.append(generate_evaluate_neg(user[-1].item(), user_dict, item_set))
        root.append(d[1]['root'].detach().numpy())

    return dgl.batch(forest), torch.LongTensor(user), torch.LongTensor(np.array(seq)), torch.LongTensor(
        target), torch.LongTensor(np.array(root)), torch.LongTensor(np.array(neg_items))


# 参照SASRec，采样100个负样本用于预测
def generate_evaluate_neg(user, user_dict, item_set, neg_num=100):
    user_neg_items = np.random.choice(np.array(list(item_set - user_dict[user])), neg_num, replace=False)

    return user_neg_items


# 传进来的参数是一个dataframe
def generate_user_dict(data, user_dict_path):
    user_dict = {}

    users = data['user'].unique().tolist()
    for user in users:
        user_item_set = set(data.loc[data['user'] == user]['item'].tolist())
        user_dict[user] = user_item_set

    with open(user_dict_path, 'wb') as f:
        pickle.dump(user_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
