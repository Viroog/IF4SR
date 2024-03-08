import argparse
import json
import pickle

import dgl
from model import IF4SR
from utils import data_partition, Sampler
import torch.optim as optim
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='grocery', help='dataset name')
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--L', type=int, default=50, help='max length of sequence')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
parser.add_argument('--hidden_unit', type=int, default=64, help='hidden dimension')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--gip_block_num', type=int, default=2, choices=[1, 2, 3],
                    help='the block num of global intention perception')
parser.add_argument('--fcb_head_num', type=int, default=2, choices=[1, 2, 4, 8],
                    help='the head num of feature capture block')
parser.add_argument('--scb_hidden_unit', type=int, default=64, help='hidden units of scb block')
parser.add_argument('--fcb_hidden_unit', type=int, default=64, help='hidden units of fcb block')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
parser.add_argument('--taxonomy_init_mode', type=str, default='default', choices=['glove', 'word2vec', 'default', 'word2vec'],
                    help='how to init taxonomy embedding')

args = parser.parse_args()

if __name__ == '__main__':

    train, valid, test, user_num, item_num = data_partition(args.dataset)
    num_batch = len(train) // args.batch_size

    taxonomy_tree_path = f'./dataset/{args.dataset}/item_taxonomy.dict'
    with open(taxonomy_tree_path, 'rb') as f:
        taxonomy_tree = pickle.load(f)

    taxonomy_path = f'./dataset/{args.dataset}/taxonomy2id.json'
    with open(taxonomy_path, 'r') as f:
        taxonomy2id = json.load(f)
    taxonomy_num = len(taxonomy2id)

    sampler = Sampler(train, taxonomy_tree, user_num, item_num, batch_size=args.batch_size, L=args.L,
                      n_workes=3)
    model = IF4SR(args, item_num, taxonomy_num).to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    model.train()

    for epoch in range(args.epoch):
        for step in range(num_batch):
            # 每个batch中的root数量都不相等，需要解决一下
            user, seq, pos, neg, root, forest = sampler.next_batch()
            user, seq, pos, neg, root = np.array(user), np.array(seq), np.array(pos), np.array(neg), np.array(root)

            batch_forest = dgl.batch(list(forest))

            pos_logit, neg_logit = model(user, seq, pos, neg, root, batch_forest)

            loss = None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()

            model.train()

    sampler.close()
