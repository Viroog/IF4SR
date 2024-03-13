import json

import numpy as np
import torch
import argparse

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='grocery', help='dataset name')
    parse.add_argument('--hidden_units', type=int, default=50, help='embedding hidden dimension')

    args = parse.parse_args()

    glove_embed_dict = {}

    with open(f'../dataset/{args.dataset}/taxonomy2id.json', 'r') as f:
        taxonomy2id = json.load(f)

    with open('./glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            splited = line.strip().split()
            word, embedding = splited[0], torch.tensor(np.asarray(splited[1:], dtype=float))
            glove_embed_dict[word] = embedding

    total_taxonomy_embed = torch.zeros((len(taxonomy2id), args.hidden_units))

    for taxonomy, taxonomy_id in taxonomy2id.items():
        # 0单独处理
        if taxonomy_id == 0:
            taxonomy_embed = torch.zeros(args.hidden_units)
        else:
            taxonomy_list = taxonomy.strip().split(' ')

            new_taxonomy_list = []
            for t in taxonomy_list:
                new_taxonomy = ''
                for c in t:
                    if c.isalpha():
                        new_taxonomy += c.lower()

                if len(new_taxonomy) > 0:
                    new_taxonomy_list.append(new_taxonomy)

            valid_cnt = 0
            taxonomy_embed = torch.zeros(args.hidden_units)
            for t in new_taxonomy_list:
                if t in glove_embed_dict.keys():
                    valid_cnt += 1
                    taxonomy_embed += glove_embed_dict[t]

            taxonomy_embed /= valid_cnt

            # 对于那些根本不存在glove中的标签，则使用xavier进行初始化
            if valid_cnt == 0:
                torch.nn.init.xavier_normal_(taxonomy_embed.unsqueeze(0))
                taxonomy_embed = taxonomy_embed.squeeze(0)

        total_taxonomy_embed[taxonomy_id, :] = taxonomy_embed

    save_path = f'../dataset/{args.dataset}/taxonomy_embed.pth'
    torch.save(total_taxonomy_embed, save_path)
