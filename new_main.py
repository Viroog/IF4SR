import argparse
import json
import multiprocessing
import os
import pickle
import time
import os
import torch

from model import IF4SR
from utils import myFloder, collate, collate_valid_test, generate_user_dict, myFloder_valid_test
from evaluate import get_ndcg_hit
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dgl import load_graphs
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='grocery', help='dataset name')
# 模型的超参数
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--L', type=int, default=50, help='max length of sequence')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
parser.add_argument('--hidden_units', type=int, default=50, help='hidden dimension')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
parser.add_argument('--device', type=str, default= 'cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda or cpu')
parser.add_argument('--taxonomy_init_mode', type=str, default='glove', choices=['glove', 'default'],
                    help='how to init taxonomy embedding')
# global intention的超参数
parser.add_argument('--block_nums', type=int, default=2, choices=[1, 2, 3],
                    help='the block num of global intention perception')
parser.add_argument('--scb_hidden_units', type=int, default=128, help='hidden units of scb block')
parser.add_argument('--fcb_hidden_units', type=int, default=128, help='hidden units of fcb block')
parser.add_argument('--global_head_nums', type=int, default=2, choices=[1, 2, 5],
                    help='the head num of feature capture block')

# local intention的超参数
parser.add_argument('--local_intention_conv', type=str, default='HGT', choices=['GAT', 'HGT', 'both'],
                    help='way of tree convolution')
parser.add_argument('--n_hop', type=int, default=2, help='gnn layer must equal to the height of tree')
parser.add_argument('--local_head_nums', type=int, default=2, help='the head num of attention mechanism in GAT/HGT')

# early stop
parser.add_argument('--max_tolerant', type=int, default=3, help='if performance dont improve for several test, stop it')

args = parser.parse_args()

if __name__ == '__main__':

    data_df = pd.read_csv(f'./dataset/{args.dataset}/{args.dataset}.txt', sep=' ', names=['user', 'item'])
    user_num = len(data_df['user'].unique())
    item_num = len(data_df['item'].unique())

    print(f'user num: {user_num}, item num: {item_num}')
    print(f'interaction num: {len(data_df)}, sparsity: {(1 - (len(data_df) / (user_num * item_num))) * 100}%')

    taxonomy_tree_path = f'./dataset/{args.dataset}/item_taxonomy.dict'
    with open(taxonomy_tree_path, 'rb') as f:
        taxonomy_tree = pickle.load(f)

    # 统计各阶的标签种类，同时方便后面的root填充
    taxonomy_cnt = {}
    for hop in range(args.n_hop):
        taxonomy_cnt[hop + 1] = set()

    for item, taxonomies in taxonomy_tree.items():
        for idx, taxonomy in enumerate(taxonomies):
            taxonomy_cnt[idx + 1].add(taxonomy)

    for hop in range(args.n_hop):
        print(f'{hop + 1} order taxonomy nums: {len(taxonomy_cnt[hop + 1])}')

    taxonomy_path = f'./dataset/{args.dataset}/taxonomy2id.json'
    with open(taxonomy_path, 'r') as f:
        taxonomy2id = json.load(f)
    # 这个taxonomy_num已经包括了0，因此在模型中的embedding不需要+1
    taxonomy_num = len(taxonomy2id)

    # 获取用户交互过的物品
    user_dict_path = f'./dataset/{args.dataset}/user_dict.dict'
    if os.path.exists(user_dict_path) is False:
        generate_user_dict(data_df, user_dict_path)
    # 读取
    with open(user_dict_path, 'rb') as f:
        user_dict = pickle.load(f)

    # 加载数据集部分代码参照DGSR源码
    train_root = f'./new_dataset/{args.dataset}_{args.L}_{args.n_hop}/train/'
    valid_root = f'./new_dataset/{args.dataset}_{args.L}_{args.n_hop}/valid/'
    test_root = f'./new_dataset/{args.dataset}_{args.L}_{args.n_hop}/test/'

    # 所有item的物品集合
    item_set = set(range(1, item_num + 1))

    train_set = myFloder(train_root, load_graphs)
    # valid_set = myFloder(valid_root, load_graphs)
    # test_set = myFloder(test_root, load_graphs)
    valid_set = myFloder_valid_test(valid_root, load_graphs, user_dict, item_set)
    test_set = myFloder_valid_test(test_root, load_graphs, user_dict, item_set)

    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                                  collate_fn=collate, shuffle=True,
                                  pin_memory=True, num_workers=8, drop_last=True)
    # val_dataloader = DataLoader(dataset=valid_set, batch_size=args.batch_size,
    #                             collate_fn=lambda x: collate_valid_test(x, user_dict, item_set),
    #                             pin_memory=True, num_workers=4)
    # test_dataloader = DataLoader(dataset=test_set, batch_size=args.batch_size,
    #                              collate_fn=lambda x: collate_valid_test(x, user_dict, item_set),
    #                              pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(dataset=valid_set, batch_size=args.batch_size,
                                collate_fn=collate_valid_test,
                                pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                                 collate_fn=collate_valid_test,
                                 pin_memory=True, num_workers=4)

    model = IF4SR(args, item_num, taxonomy_num, len(taxonomy_cnt[1])).to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_ndcg, best_hit = 0.0, 0.0
    best_model = None
    now_tolerant = 0

    model.train()
    for epoch in range(args.epoch):
        epoch_loss = 0
        t1 = time.time()
        for batch_data in train_dataloader:
            forest, user, seq, target, root = batch_data
            score = model(seq, root, forest, rec_items=None, training=True)

            loss = criterion(score, target.to(args.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        t2 = time.time()
        print(f'epoch: {epoch + 1}, cost_time: {t2 - t1}, loss: {epoch_loss / len(train_dataloader)}')

        if (epoch + 1) % 20 == 0:
            model.eval()

            # 预测不保留梯度，减少负载
            with torch.no_grad():
                total_rec_score = []
                for batch_data in val_dataloader:
                    forest, user, seq, target, root, neg_items = batch_data
                    rec_items = torch.concat([target.unsqueeze(-1), neg_items], dim=-1)
                    score, rec_score = model(seq, root, forest, rec_items=rec_items, training=False)
                    total_rec_score.append(rec_score.detach().cpu().numpy())
                valid_ndcg_10, valid_hit_10, valid_ndcg_20, valid_hit_20 = get_ndcg_hit(total_rec_score)

                total_rec_score = []
                for batch_data in test_dataloader:
                    forest, user, seq, target, root, neg_items = batch_data
                    rec_items = torch.concat([target.unsqueeze(-1), neg_items], dim=-1)
                    score, rec_score = model(seq, root, forest, rec_items=rec_items, training=False)
                    total_rec_score.append(rec_score.detach().cpu().numpy())
                test_ndcg_10, test_hit_10, test_ndcg_20, test_hit_20 = get_ndcg_hit(total_rec_score)

                print(
                    f'valid (NDCG@10: {valid_ndcg_10}, HIT@10: {valid_hit_10}, NDCG@20: {valid_ndcg_20}, HIT@20: {valid_hit_20})')
                print(
                    f'test (NDCG@10: {test_ndcg_10}, HIT@10: {test_hit_10}, NDCG@20: {test_ndcg_20}, HIT@20: {test_hit_20})')

            if test_ndcg_10 > best_ndcg or test_hit_10 > best_hit:
                best_ndcg, best_hit = test_ndcg_10, test_hit_10
                best_model = model
                now_tolerant = 0

                model_path = f'trained/{args.dataset}/model.pth'
                torch.save(best_model, model_path)
            else:
                now_tolerant += 1

            if now_tolerant > args.max_tolerant:
                break

            model.train()
