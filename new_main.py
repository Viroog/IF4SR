import argparse
import json
import pickle
import time

import dgl
import torch

from model import IF4SR
from utils import data_partition, myFloder, collate, collate_valid_test
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dgl import load_graphs

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='grocery', help='dataset name')
# 模型的超参数
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--L', type=int, default=50, help='max length of sequence')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
parser.add_argument('--hidden_units', type=int, default=50, help='hidden dimension')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
parser.add_argument('--taxonomy_init_mode', type=str, default='glove', choices=['glove', 'default'],
                    help='how to init taxonomy embedding')
# global intention的超参数
parser.add_argument('--gip_block_nums', type=int, default=2, choices=[1, 2, 3],
                    help='the block num of global intention perception')
parser.add_argument('--fcb_head_nums', type=int, default=2, choices=[1, 2, 5],
                    help='the head num of feature capture block')
parser.add_argument('--scb_hidden_units', type=int, default=64, help='hidden units of scb block')
parser.add_argument('--fcb_hidden_units', type=int, default=64, help='hidden units of fcb block')

# local intention的超参数
parser.add_argument('--gnn_conv', type=str, default='GAT', choices=['GAT', 'HGT'], help='way of tree convolution')
parser.add_argument('--gnn_layers', type=int, default=2, help='gnn layer must equal to the height of tree')
parser.add_argument('--gnn_head_nums', type=int, default=2, help='the head num of attention mechanism in GAT/HGT')

# early stop
parser.add_argument('--max_tolerant', type=int, default=3, help='if performance dont improve for several test, stop it')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    train, valid, test, user_num, item_num = dataset

    taxonomy_tree_path = f'./dataset/{args.dataset}/item_taxonomy.dict'
    with open(taxonomy_tree_path, 'rb') as f:
        taxonomy_tree = pickle.load(f)

    # 统计各阶的标签种类，同时方便后面的root填充
    taxonomy_cnt = {}
    for hop in range(args.gnn_layers):
        taxonomy_cnt[hop + 1] = set()

    for item, taxonomies in taxonomy_tree.items():
        for idx, taxonomy in enumerate(taxonomies):
            taxonomy_cnt[idx + 1].add(taxonomy)

    for hop in range(args.gnn_layers):
        print(f'{hop + 1} order taxonomy nums: {len(taxonomy_cnt[hop + 1])}')

    taxonomy_path = f'./dataset/{args.dataset}/taxonomy2id.json'
    with open(taxonomy_path, 'r') as f:
        taxonomy2id = json.load(f)
    # 这个taxonomy_num已经包括了0，因此在模型中的embedding不需要+1
    taxonomy_num = len(taxonomy2id)

    # 加载数据集
    train_root = f'./new_dataset/{args.dataset}/train/'
    valid_root = f'./new_dataset/{args.dataset}/valid/'
    test_root = f'./new_dataset/{args.dataset}/test/'

    train_set = myFloder(train_root, load_graphs, train, item_num)
    valid_set = myFloder(valid_root, load_graphs, train, item_num)
    test_set = myFloder(test_root, load_graphs, train, item_num)

    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=collate, shuffle=True,
                                  pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(dataset=valid_set, batch_size=args.batch_size, collate_fn=collate_valid_test,
                                pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=collate_valid_test,
                                 pin_memory=True, num_workers=4)

    model = IF4SR(args, item_num, taxonomy_num, len(taxonomy_cnt[1])).to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    model.train()

    best_ndcg, best_hit = 0.0, 0.0
    best_model = None
    now_tolerant = 0

    for epoch in range(args.epoch):
        epoch_loss = 0
        t1 = time.time()
        for idx, batch_data in enumerate(train_dataloader):
            forest, user, seq, pos, neg, root = batch_data

            pos_logit, neg_logit = model(seq, pos, neg, root, forest)

            pos_label, neg_label = torch.ones(pos_logit.shape, device=args.device), torch.zeros(neg_logit.shape,
                                                                                                device=args.device)

            loss = criterion(pos_logit, pos_label)
            loss += criterion(neg_logit, neg_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        t2 = time.time()
        print(f'epoch: {epoch + 1}, cost_time: {t2 - t1}, loss: {epoch_loss / len(train_dataloader)}')

        if (epoch + 1) % 20 == 0:
            model.eval()
            # 验证集
            for idx, batch_data in enumerate(val_dataloader):
                valid_ndcg, valid_hit = None, None

            # 测试集
            for idx, batch_data in enumerate(test_dataloader):
                test_ndcg, test_hit = None, None

            print(f'epoch: {epoch + 1}, valid (NDCG@10: {valid_ndcg}, HIT@10: {valid_hit}), test (NDCG@10: {test_ndcg}, HIT@10: {test_hit})')

            # 这里没写完
            if test_ndcg > best_ndcg or test_hit > best_hit:
                best_ndcg, best_hit = test_ndcg, test_hit
                best_model = model
                now_tolerant = 0

                model_path = f'trained/{args.dataset}/model.pth'
                torch.save(best_model, model_path)

            else:
                now_tolerant += 1

            if now_tolerant > args.max_tolerant:
                break

            print(
                f'epoch: {epoch + 1}, valid (NDCG@10:{valid_ndcg}, HIT@10:{valid_hit}), test (NDCG@10:{test_ndcg}, HIT@10: {test_hit})')

            model.train()
