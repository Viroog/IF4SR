import argparse

import torch
from dgl import save_graphs
import pickle
import dgl
import pandas as pd
from joblib import Parallel, delayed


# 填充用户交互序列至最大长度(args.L)
def pad_seq(u_seq, max_len, pad):
    # 大于最大长度，选最后L个
    if len(u_seq) >= max_len:
        pad_u_seq = u_seq[-max_len:]
    # 小于最大长度，在前面补0
    else:
        pad_u_seq = [pad] * (max_len - len(u_seq)) + u_seq

    return pad_u_seq


# 填充根节点至最大长度(first_taxonomy_num)
def pad_root(u_root, max_root_num, pad):
    if len(u_root) < max_root_num:
        pad_u_root = u_root + [pad] * (max_root_num - len(u_root))
        return pad_u_root


def generate_user_intent_forest(u_seq):
    # 根节点绝对id，即在整个taxonomy中的id
    absolute_root = set()

    # 物品及物品的父类标签(也是最后一个级别的标签)
    items, items_parent = [], []
    # 子类标签和父类标签
    t_childs, t_parents = [], []

    for item in u_seq:
        # taxonomies: [一级标签，二级标签，...]
        taxonomies = taxonomy_tree[item]

        items.append(item)
        items_parent.append(taxonomies[-1])

        # 从后往前遍历
        idx = len(taxonomies) - 1
        while idx > 0:
            t_childs.append(taxonomies[idx])
            t_parents.append(taxonomies[idx - 1])

            idx -= 1
            # 当idx等于0时，指向树的根节点
            if idx == 0:
                absolute_root.add(taxonomies[idx])

    # 构建森林的数据
    forest_data = {
        ('item', 'i2t', 'taxonomy'): (torch.LongTensor(items), torch.LongTensor(items_parent)),
        ('taxonomy', 't2t', 'taxonomy'): (torch.LongTensor(t_childs), torch.LongTensor(t_parents))
    }

    # 构建森林
    forest = dgl.heterograph(forest_data)

    max_item_id, max_taxonomy_id = max(set(items)), max(set(t_childs + t_parents))

    # 赋予节点id数据，后面在GNN中会用到
    # 左闭右开
    forest.nodes['item'].data['id'] = torch.LongTensor(list(range(0, max_item_id + 1)))
    forest.nodes['taxonomy'].data['id'] = torch.LongTensor(list(range(0, max_taxonomy_id + 1)))

    # 提取子图，去除不必要的节点
    forest = dgl.node_subgraph(forest, {'item': list(set(items)), 'taxonomy': list(set(t_childs + t_parents))})

    # 获取根节点的相对id(即在图中从0开始排序的id)，后面能够加快模型训练速度
    relative_root = torch.where(
        torch.isin(forest.nodes['taxonomy'].data['id'], torch.LongTensor(list(absolute_root))))[0].tolist()
    # 使用-1的原因是，0已经被relative_root占用了
    relative_root = pad_root(relative_root, len(first_taxonomy_set), -1)

    return forest, relative_root


# 在将用户的序列分割的同时，生成意图森林
def generate_user(u):
    # 用户u的交互序列(原数据已经按时间排序好)
    u_seq = data_df.loc[data_df['user'] == u]['item'].values.tolist()
    # 序列终点
    end = len(u_seq) - 1

    # 长度小于4无法生成验证集和测试集
    # 训练集长度至少为2(一个训练数据+一个标签)，验证集和测试集取最后两个
    if len(u_seq) < 4:
        return 0, 0
    else:
        # u_seq[t]作为target
        # 左闭右开 t∈[1,2,...,len(u_seq)-1]
        for t in range(1, len(u_seq)):
            # u_seq_t={[i_1],[i_1,i_2]...,[i_1,i_2,...,i_len(u_seq)-2]}
            u_seq_t = u_seq[0:t]
            forest, root = generate_user_intent_forest(u_seq_t)
            pad_u_seq_t = pad_seq(u_seq_t, args.L, 0)
            # 目标
            target = u_seq[t]

            # 当t<end-1时，是训练集
            if t < end - 1:
                save_graphs(f'{train_path}/{u}/{u}_{t}.bin', forest,
                            {'user': torch.LongTensor([u]), 'seq': torch.LongTensor(pad_u_seq_t),
                             'target': torch.LongTensor([target]), 'root': torch.LongTensor(root)})
            # 当t指向倒数第二个物品，是验证集
            elif t == end - 1:
                save_graphs(f'{valid_path}/{u}/{u}_{t}.bin', forest,
                            {'user': torch.LongTensor([u]), 'seq': torch.LongTensor(pad_u_seq_t),
                             'target': torch.LongTensor([target]), 'root': torch.LongTensor(root)})
            # 当t指向最后一个物品，是测试集
            elif t == end:
                save_graphs(f'{test_path}/{u}/{u}_{t}.bin', forest,
                            {'user': torch.LongTensor([u]), 'seq': torch.LongTensor(pad_u_seq_t),
                             'target': torch.LongTensor([target]), 'root': torch.LongTensor(root)})


# 参照DGSR论文的源码
def generate_data(job=10):
    user = data_df['user'].unique()
    Parallel(n_jobs=job)(delayed(lambda u: generate_user(u))(u) for u in user)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='grocery')
    parse.add_argument('--job', type=int, default=10, help='parallel job num')
    parse.add_argument('--n_hop', type=int, default=2, help='category layer num')
    parse.add_argument('--L', type=int, default=50, help='max length of user interact sequence')

    args = parse.parse_args()

    data_path = f'dataset/{args.dataset}/{args.dataset}.txt'
    data_df = pd.read_csv(f'./dataset/{args.dataset}/{args.dataset}.txt', sep=' ', names=['user', 'item'])

    train_path = f'new_dataset/{args.dataset}_{args.L}_{args.n_hop}/train/'
    valid_path = f'new_dataset/{args.dataset}_{args.L}_{args.n_hop}/valid/'
    test_path = f'new_dataset/{args.dataset}_{args.L}_{args.n_hop}/test/'

    taxonomy_tree_path = f'dataset/{args.dataset}/item_taxonomy.dict'
    with open(taxonomy_tree_path, 'rb') as f:
        taxonomy_tree = pickle.load(f)

    first_taxonomy_set = set()

    for item, taxonomies in taxonomy_tree.items():
        for idx, taxonomy in enumerate(taxonomies):
            if idx == 0:
                first_taxonomy_set.add(taxonomy)

    generate_data(job=args.job)
