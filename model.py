import time

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


# 将返回结果传进来，逆向执行unbatch操作，还原节点id来代替unbatch操作
def graph_taxonomy(forest, root, taxonomy_embed):
    b_taxonomy_size = forest.batch_num_nodes('taxonomy')

    # roll: (inputs, shifts, dims=None) 往指定方向位移，如果dims为None，则会flatten到一维再变回原来的形状
    # cumsum: 求前缀和
    tmp = torch.roll(torch.cumsum(b_taxonomy_size, 0), 1)
    tmp[0] = 0
    # 上述所有操作执行完后，tmp[i]~tmp[i+1]属于第i个batch的节点编号(0 <= i < batch, i∈N)

    tt = torch.tile(tmp.unsqueeze(-1), dims=(1, root.shape[1]))
    root_idx = root + torch.tile(tmp.unsqueeze(-1), dims=(1, root.shape[1]))

    return root_idx


# 异构图之间也是调用GAT卷积
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GATLayer, self).__init__()

        # aggregate参数指定与否都没有太大关系，因为每个dst node只有一种src node
        self.conv = dglnn.HeteroGraphConv({
            'i2t': dglnn.GATConv(in_feats, out_feats, num_heads=num_heads),
            't2t': dglnn.GATConv(in_feats, out_feats, num_heads=num_heads)
        })

    def forward(self, g, feat):
        rsts = self.conv(g, feat)

        return rsts


class HGTLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(HGTLayer, self).__init__()

        self.conv = dglnn.HeteroGraphConv({
            'i2t': dglnn.HGTConv(),
            't2t': dglnn.HGTConv()
        })

    def forward(self):
        pass


class IF4SR(nn.Module):
    def __init__(self, args, item_num, taxonomy_num, first_taxonomy_num):
        super(IF4SR, self).__init__()

        self.item_embed = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        # 标签单独初始化
        self.taxonomy_embed = nn.Embedding(taxonomy_num, args.hidden_units, padding_idx=0)
        self.first_taxonomy_num = first_taxonomy_num

        self.dropout = nn.Dropout(p=args.dropout_rate)

        # sequence capture block
        self.scb_layernorms = nn.ModuleList()
        self.scb_transform1 = nn.ModuleList()
        self.scb_transform2 = nn.ModuleList()
        # feature capture block
        self.fcb_layernorms = torch.nn.ModuleList()
        self.fcb_transform1 = nn.ModuleList()
        self.fcb_transform2 = nn.ModuleList()
        self.fcb_transform3 = nn.ModuleList()

        self.weighted_vector = nn.Linear(args.hidden_units, 1, bias=False)

        for _ in range(args.gip_block_nums):
            # scb
            self.scb_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            # transform_matrix1: (hidden_unit, L)
            self.scb_transform1.append(nn.Linear(args.L, args.scb_hidden_units, bias=False))
            # transform_matrix2: (L, hidden_units)
            self.scb_transform2.append(nn.Linear(args.scb_hidden_units, args.L, bias=False))
            # fcb
            self.fcb_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.fcb_transform1.append(
                nn.Linear(int(args.hidden_units / args.fcb_head_nums), args.fcb_hidden_units, bias=False))
            self.fcb_transform2.append(
                nn.Linear(args.fcb_hidden_units, int(args.hidden_units / args.fcb_head_nums), bias=False))
            self.fcb_transform3.append(nn.Linear(args.hidden_units, args.hidden_units, bias=False))

        self.gnn_layer = nn.ModuleList()

        for _ in range(args.gnn_layers):
            if args.gnn_conv == 'GAT':
                new_gnn_layer = GATLayer(args.hidden_units, int(args.hidden_units / args.gnn_head_nums),
                                         args.gnn_head_nums)
            elif args.gnn_conv == 'HGT':
                new_gnn_layer = HGTLayer()

            self.gnn_layer.append(new_gnn_layer)

        self.args = args
        self.init_params()

    # 初始化参数
    def init_params(self):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
                # 单独初始化
                if name == 'taxonomy_embed.weight':
                    if self.args.taxonomy_init_mode == 'default':
                        pass
                    elif self.args.taxonomy_init_mode == 'glove':
                        embed_path = f'./dataset/{self.args.dataset}/taxonomy_embed.pth'
                        embed_data = torch.load(embed_path)
                        self.taxonomy_embed.weight.data = embed_data
            except:
                pass

    # 根节点root用于获取最后聚合完的K个意图向量
    # 森林forest本质上是一个非连通图，有K个非联通图。并且每个图都是树形结构
    def get_local_intention(self, root, forest):

        forest = forest.to(self.args.device)
        # root = torch.LongTensor(root).to(self.args.device)
        root = root.to(self.args.device)

        forest.nodes['item'].data['h'] = self.item_embed(forest.nodes['item'].data['id'].to(self.args.device))
        forest.nodes['taxonomy'].data['h'] = self.taxonomy_embed(
            forest.nodes['taxonomy'].data['id'].to(self.args.device))

        for layer in range(self.args.gnn_layers):
            feat = {
                'item': forest.nodes['item'].data['h'],
                'taxonomy': forest.nodes['taxonomy'].data['h']
            }
            # 只是返回了结果，图中的节点数据并没有被更新
            rsts = self.gnn_layer[layer].forward(forest, feat)

            # 更新图中的节点数据
            forest.nodes['taxonomy'].data['h'] = rsts['taxonomy'].view(rsts['taxonomy'].shape[0], -1)

        # 要不要将每层的根节向量记录起来后累加 待定

        # 反转batch操作
        # 这里不能用unbatch操作，非常耗时， 可以直接使用gnn前向传播返回的结果来计算
        # unbatch_forest = dgl.unbatch(forest)

        # 找出root中非padding的位置
        valid_root_mask = torch.where(root != -1)

        # n个自注意力头，(N, n, args.hidden_units)
        taxonomy_embed = rsts['taxonomy'].view(rsts['taxonomy'].shape[0], -1)
        # 获得根节点在taxonomy_embed中的位置
        root_idx = graph_taxonomy(forest, root, taxonomy_embed)

        # 根据根节点获得其对应的embed
        valid_root_embed = taxonomy_embed[root_idx[valid_root_mask]]

        # 局部向量
        # 这里要用forest的batch_size，而不是整个batch大小
        local_intention = torch.zeros((forest.batch_size, self.first_taxonomy_num, self.args.hidden_units),
                                      device=self.args.device)
        # 将值填入局部向量中
        local_intention[valid_root_mask[0], valid_root_mask[1], :] = valid_root_embed

        # for i in range(len(unbatch_forest)):
        #     # indices = torch.where(torch.isin(unbatch_forest[i].nodes['taxonomy'].data['id'],
        #     #                                  root[i].to(self.args.device)))
        #     indices = torch.where(torch.isin(unbatch_forest[i].nodes['taxonomy'].data['id'],
        #                                      torch.LongTensor(root[i]).to(self.args.device)))
        #
        #     user_local_intention = unbatch_forest[i].nodes['taxonomy'].data['h'][indices]
        #
        #     local_intention[i, :user_local_intention.shape[0], :] = user_local_intention

        return local_intention

    def get_global_intention(self, seq):
        # item embedding: (batch_size, L, hidden_unit) 第0层
        # V = self.item_embed(torch.LongTensor(seq).to(self.args.device))
        V = self.item_embed(seq.to(self.args.device))

        for i in range(self.args.gip_block_nums):
            # scb
            normed_V = self.scb_layernorms[i](V)
            transposed_normed_V = torch.transpose(normed_V, 1, 2)
            scb_out = torch.transpose(
                self.scb_transform2[i](F.gelu(self.scb_transform1[i](transposed_normed_V))), 1, 2)
            V_scb = normed_V + scb_out

            # fcb
            normed_V_scb = self.fcb_layernorms[i](V_scb)
            concated = None
            for h in range(self.args.fcb_head_nums):
                # 右边界不包括
                start, end = int(h * (self.args.hidden_units / self.args.fcb_head_nums)), int(
                    (h + 1) * (self.args.hidden_units / self.args.fcb_head_nums))
                partial = normed_V_scb[:, :, start:end]

                headn_out = self.fcb_transform2[i](F.gelu(self.fcb_transform1[i](partial)))

                if concated is None:
                    concated = headn_out
                else:
                    concated = torch.concat([concated, headn_out], dim=-1)

            fcb_out = self.fcb_transform3[i](concated)

            V = normed_V_scb + fcb_out

        # 现在的V为经过N个(SCB+FCB)模块后得到的物品表征向量
        weighted_alpha = F.softmax(self.weighted_vector(V), dim=1)
        # 对于序列长度维度进行求和
        # (batch_size, hidden_unit)
        global_intention = self.dropout(torch.sum(weighted_alpha * V, dim=1))

        return global_intention

    def get_intention(self, global_intention, local_intention):
        # 填充项不能参加softmax函数，影响了权重
        # 创建掩码，将不参加的位置设置为负无穷，负无穷在softmax中不参与计算
        # mul_res = torch.sum(global_intention.unsqueeze(dim=1) * local_intention, dim=-1)
        mul_res = local_intention.matmul(global_intention.unsqueeze(dim=-1)).squeeze(-1)
        mask = (mul_res != 0).float()
        masked_mul_res = torch.where(mask != 0, mul_res, float('-inf'))

        local_intention_weight = F.softmax(masked_mul_res, dim=-1)

        intention = global_intention + torch.sum(local_intention_weight.unsqueeze(dim=-1) * local_intention, dim=1)

        return intention

    def forward(self, seq, pos, neg, root, forest):

        # (batch_size, hidden_units)
        global_intention = self.get_global_intention(seq)
        # (batch_size, K, hidden_units): K为根节点数量
        local_intention = self.get_local_intention(root, forest)

        intention = self.get_intention(global_intention, local_intention)

        pos_embed = self.item_embed(pos.to(self.args.device))
        neg_embed = self.item_embed(neg.to(self.args.device))
        # pos_embed = self.item_embed(torch.LongTensor(pos).to(self.args.device))
        # neg_embed = self.item_embed(torch.LongTensor(neg).to(self.args.device))

        # 与正负样本预测评分
        pos_logit = torch.sum(intention * pos_embed, dim=-1)
        neg_logit = torch.sum(intention * neg_embed, dim=-1)

        return pos_logit, neg_logit

    def predict(self, seq, items, root, forest):
        global_intention = self.get_global_intention(seq)
        local_intention = self.get_local_intention(root, forest)

        # batch_size = 1
        # (batch_size, hidden_units)
        intention = self.get_intention(global_intention, local_intention)

        # (batch_size, len(items), hidden_units)
        item_embed = self.item_embed(torch.LongTensor(items).to(self.args.device))

        logit = item_embed.matmul(intention.unsqueeze(-1)).squeeze(-1)

        return logit
