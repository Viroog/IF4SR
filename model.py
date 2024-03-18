import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


# 将返回结果传进来，逆向执行unbatch操作，还原节点id来代替unbatch操作
def graph_taxonomy(forest, root):
    batch_taxonomy_size = forest.batch_num_nodes('taxonomy')

    # roll: (inputs, shifts, dims=None) 往指定方向位移，如果dims为None，则会flatten到一维再变回原来的形状
    # cumsum: 求前缀和
    tmp = torch.roll(torch.cumsum(batch_taxonomy_size, 0), 1)
    tmp[0] = 0
    # 上述所有操作执行完后，tmp[i]~tmp[i+1]属于第i个batch的节点编号(0 <= i < batch, i∈N)
    root_idx = root + torch.tile(tmp.unsqueeze(-1), dims=(1, root.shape[1]))

    return root_idx


# 异构图之间也是调用GAT卷积
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout_rate):
        super(GATLayer, self).__init__()

        # aggregate参数指定与否都没有太大关系，因为每个dst node只有一种src node
        self.conv = dglnn.HeteroGraphConv({
            'i2t': dglnn.GATConv(in_feats, out_feats, num_heads=num_heads, feat_drop=dropout_rate, residual=True,
                                 activation=nn.ReLU()),
            't2t': dglnn.GATConv(in_feats, out_feats, num_heads=num_heads, feat_drop=dropout_rate, residual=True,
                                 activation=nn.ReLU())
        })

    def forward(self, g, feat):
        rsts = self.conv(g, feat)

        return rsts


class HGTLayer(nn.Module):
    def __init__(self, in_feats, head_size, num_heads, num_ntypes=2, num_etypes=2, dropout_rate=0.2):
        super(HGTLayer, self).__init__()

        self.conv = dglnn.HGTConv(in_size=in_feats, head_size=head_size, num_heads=num_heads, num_ntypes=num_ntypes,
                                  num_etypes=num_etypes, dropout=dropout_rate)

    def forward(self, g, feat):
        rsts = self.conv(g, feat)

        return rsts


# 很多linear没有去bias，和论文中写的并不太一样
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.gelu(self.linear1(x)))
        x = self.dropout(self.linear2(x))

        return x


class Mixer_MLP(nn.Module):
    def __init__(self, L, hidden_units, scb_hidden_units, fcb_hidden_units, dropout_rate, head_nums):
        super(Mixer_MLP, self).__init__()

        self.head_nums = head_nums

        self.scb = FeedForward(input_dim=L, hidden_dim=scb_hidden_units, dropout_rate=dropout_rate)
        self.scb_layer_norm = nn.LayerNorm(hidden_units)

        self.fcb = nn.ModuleList()
        for head in range(head_nums):
            new_fcb_head = FeedForward(input_dim=int(hidden_units / head_nums), hidden_dim=fcb_hidden_units,
                                       dropout_rate=dropout_rate)
            self.fcb.append(new_fcb_head)

        self.fcb_layer_norm = nn.LayerNorm(hidden_units)
        self.fcb_fuse_head = nn.Linear(hidden_units, hidden_units)


    def fcb_forward(self, x):
        x = self.fcb_layer_norm(x)
        x_list = torch.split(x, int(x.shape[-1] / self.head_nums), dim=-1)
        head_out = [fcb_chunk(x_list[i]) for i, fcb_chunk in enumerate(self.fcb)]
        x = torch.concat(head_out, dim=-1)
        x = self.fcb_fuse_head(x)

        return x

    def scb_forward(self, x):
        x = self.scb_layer_norm(x)
        # 转置
        x = x.transpose(2, 1)
        x = self.scb(x)
        # 转置回来
        x = x.transpose(1, 2)

        return x

    def forward(self, x):
        # sequential capture block
        x = x + self.scb_forward(x)

        # feature capture block
        x = x + self.fcb_forward(x)

        return x


class IF4SR(nn.Module):
    def __init__(self, args, item_num, taxonomy_num, first_taxonomy_num):
        super(IF4SR, self).__init__()

        self.item_embed = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        # 标签单独初始化
        self.taxonomy_embed = nn.Embedding(taxonomy_num, args.hidden_units, padding_idx=0)
        self.first_taxonomy_num = first_taxonomy_num

        self.dropout = nn.Dropout(p=args.dropout_rate)

        # # sequence capture block
        # self.scb_layernorms = nn.ModuleList()
        # self.scb_transform1 = nn.ModuleList()
        # self.scb_transform2 = nn.ModuleList()
        # # feature capture block
        # self.fcb_layernorms = torch.nn.ModuleList()
        # self.fcb_transform1 = nn.ModuleList()
        # self.fcb_transform2 = nn.ModuleList()
        # self.fcb_transform3 = nn.ModuleList()
        #
        # self.global_intention_weight = nn.Linear(args.hidden_units, 1, bias=False)
        #
        # for _ in range(args.block_nums):
        #     # scb
        #     self.scb_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
        #     # transform_matrix1: (hidden_unit, L)
        #     self.scb_transform1.append(nn.Linear(args.L, args.scb_hidden_units, bias=False))
        #     # transform_matrix2: (L, hidden_units)
        #     self.scb_transform2.append(nn.Linear(args.scb_hidden_units, args.L, bias=False))
        #     # fcb
        #     self.fcb_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
        #     self.fcb_transform1.append(
        #         nn.Linear(int(args.hidden_units / args.fcb_head_nums), args.fcb_hidden_units, bias=False))
        #     self.fcb_transform2.append(
        #         nn.Linear(args.fcb_hidden_units, int(args.hidden_units / args.fcb_head_nums), bias=False))
        #     self.fcb_transform3.append(nn.Linear(args.hidden_units, args.hidden_units, bias=False))

        # 全局意图
        self.global_intention_layers = nn.ModuleList()
        # MLP4Rec(global)
        for _ in range(args.block_nums):
            new_global_intention_layer = Mixer_MLP(args.L, args.hidden_units, args.scb_hidden_units,
                                                   args.fcb_hidden_units, args.dropout_rate, args.global_head_nums)
            self.global_intention_layers.append(new_global_intention_layer)
        self.global_intention_attn = nn.Linear(args.hidden_units, 1, bias=False)

        # 局部意图
        self.local_intention_layers = nn.ModuleList()

        for _ in range(args.n_hop):
            if args.local_intention_conv == 'GAT':
                new_local_intention_layer = GATLayer(args.hidden_units, int(args.hidden_units / args.local_head_nums),
                                                     args.local_head_nums, args.dropout_rate)
            elif args.local_intention_conv == 'HGT':
                new_local_intention_layer = HGTLayer(in_feats=args.hidden_units,
                                                     head_size=int(args.hidden_units / args.local_head_nums),
                                                     num_heads=args.local_head_nums, dropout_rate=args.dropout_rate)

            self.local_intention_layers.append(new_local_intention_layer)

        self.local_intention_attn_norm = nn.LayerNorm(first_taxonomy_num, eps=1e-8)
        # 全局
        self.intention_norm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.unified_map = nn.Linear(args.hidden_units, args.hidden_units, bias=False)

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

    # 局部意图的变量也重命名一下，也很丑陋
    def get_local_intention(self, root, forest):

        forest = forest.to(self.args.device)
        root = root.to(self.args.device)

        forest.nodes['item'].data['h'] = self.item_embed(forest.nodes['item'].data['id'].to(self.args.device))
        forest.nodes['taxonomy'].data['h'] = self.taxonomy_embed(
            forest.nodes['taxonomy'].data['id'].to(self.args.device))

        # 记录每层得到的结果
        k_list = []
        for layer in range(self.args.n_hop):
            feat = {
                'item': forest.nodes['item'].data['h'],
                'taxonomy': forest.nodes['taxonomy'].data['h']
            }
            # 只是返回了结果，图中的节点数据并没有被更新
            rsts = self.local_intention_layers[layer].forward(forest, feat)

            taxonomy_update_embed = rsts['taxonomy'].view(rsts['taxonomy'].shape[0], -1)
            # 更新图中的节点数据
            forest.nodes['taxonomy'].data['h'] = taxonomy_update_embed
            k_list.append(taxonomy_update_embed)

        # 反转batch操作
        # 这里不能用unbatch操作，非常耗时， 可以直接使用gnn前向传播返回的结果来计算
        # unbatch_forest = dgl.unbatch(forest)

        # 找出root中非padding的位置
        valid_root_mask = torch.where(root != -1)

        # 获得根节点在taxonomy_embed中的位置
        root_idx = graph_taxonomy(forest, root)

        # 累加
        total_k_taxonomy_embed = None
        for k in k_list:
            if total_k_taxonomy_embed is None:
                total_k_taxonomy_embed = k
            else:
                total_k_taxonomy_embed = k + total_k_taxonomy_embed

        # 求均值
        total_k_taxonomy_embed = total_k_taxonomy_embed / len(k_list)

        # 根据根节点获得其对应的embed
        # valid_root_embed = taxonomy_embed[root_idx[valid_root_mask]]
        valid_root_embed = total_k_taxonomy_embed[root_idx[valid_root_mask]]

        # 局部向量
        # 这里要用forest的batch_size，而不是整个batch大小
        local_intention = torch.zeros((forest.batch_size, self.first_taxonomy_num, self.args.hidden_units),
                                      device=self.args.device)
        # 将值填入局部向量中
        local_intention[valid_root_mask[0], valid_root_mask[1], :] = valid_root_embed

        return local_intention

    def get_global_intention(self, seq):
        # item embedding: (batch_size, L, hidden_unit) 第0层
        V = self.item_embed(seq.to(self.args.device))

        # for i in range(self.args.gip_block_nums):
        #     # scb
        #     normed_V = self.scb_layernorms[i](V)
        #     transposed_normed_V = torch.transpose(normed_V, 1, 2)
        #     scb_out = torch.transpose(
        #         self.scb_transform2[i](F.gelu(self.scb_transform1[i](transposed_normed_V))), 1, 2)
        #     V_scb = normed_V + scb_out
        #
        #     # fcb
        #     normed_V_scb = self.fcb_layernorms[i](V_scb)
        #     concated = None
        #     for h in range(self.args.fcb_head_nums):
        #         # 右边界不包括
        #         start, end = int(h * (self.args.hidden_units / self.args.fcb_head_nums)), int(
        #             (h + 1) * (self.args.hidden_units / self.args.fcb_head_nums))
        #         partial = normed_V_scb[:, :, start:end]
        #
        #         headn_out = self.fcb_transform2[i](F.gelu(self.fcb_transform1[i](partial)))
        #
        #         if concated is None:
        #             concated = headn_out
        #         else:
        #             concated = torch.concat([concated, headn_out], dim=-1)
        #
        #     fcb_out = self.fcb_transform3[i](concated)
        #
        #     V = normed_V_scb + fcb_out

        for i in range(self.args.block_nums):
            V = self.global_intention_layers[i](V)

        # 没有去除padding的attn
        attn = self.global_intention_attn(V).squeeze(-1)
        # 将padding值设置为负无穷，在参与softmax计算时结果为0
        attn_mask = (seq != 0).to(self.args.device)
        # 去除padding的attn
        attn = torch.where(attn_mask, attn, float('-inf'))
        # 再经过softmax变成真正的权重
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)

        # 对序列长度维度进行求和，获得所有物品共同表示的一个全局意图
        global_intention = self.dropout(torch.sum(attn * V, dim=1))
        # 现在的V为经过N个(SCB+FCB)模块后得到的物品表征向量
        # weighted_alpha = F.softmax(self.global_intention_weight(V), dim=1)
        # 对于序列长度维度进行求和
        # (batch_size, hidden_unit)
        # global_intention = self.dropout(torch.sum(weighted_alpha * V, dim=1))

        return global_intention

    def get_intention(self, global_intention, local_intention):
        # 填充项不能参加softmax函数，影响了权重
        # 创建掩码，将不参加的位置设置为负无穷，负无穷在softmax中不参与计算
        # mul_res = torch.sum(global_intention.unsqueeze(dim=1) * local_intention, dim=-1)
        mul_res = local_intention.matmul(global_intention.unsqueeze(dim=-1)).squeeze(-1)

        # 全局意图和局部意图之间的权重(没有去除padding)
        # (batch_size, K)
        attn = local_intention.matmul(global_intention.unsqueeze(dim=-1)).squeeze(-1)
        # mask = (mul_res != 0).float()
        attn_mask = (attn != 0)
        # 将padding填充为负无穷
        attn = torch.where(attn_mask, attn, float('-inf'))
        # 经过softmax变成真正的权重
        attn = F.softmax(attn, dim=-1)

        intention = global_intention + torch.sum(attn.unsqueeze(-1) * local_intention, dim=1)

        # 最后在经过norm和dropout得到最终的intention
        intention = self.dropout(self.intention_norm(intention))

        return intention

        # attn = self.local_intention_attn_norm(attn)
        # 加一层layer norm，防止某些维度过大
        # mul_res = self.local_intention_attn_norm()
        # mask = (mul_res != 0).float()
        # masked_mul_res = torch.where(mask != 0, mul_res, float('-inf'))

        # local_intention_weight = F.softmax(masked_mul_res, dim=-1)

        # intention = global_intention + torch.sum(local_intention_weight.unsqueeze(dim=-1) * local_intention, dim=1)

        # 最后加层layer_norm以及dropout，方便梯度反向传播
        # intention = self.dropout(self.intention_norm(intention))
        #
        # return intention

    def forward(self, seq, root, forest, rec_items=None, training=False):

        # (batch_size, hidden_units)
        global_intention = self.get_global_intention(seq)
        # (batch_size, K, hidden_units): K为根节点数量，默认填充为了最大根节点数
        local_intention = self.get_local_intention(root, forest)

        # (batch_size, hidden_units)
        intention = self.get_intention(global_intention, local_intention)

        unified_intention = self.unified_map(intention)
        # 这里不排除padding0，在crossentropy损失函数中可以指定ignore_idx
        score = torch.matmul(unified_intention, self.item_embed.weight.transpose(1, 0))

        if training:
            return score
        else:
            # (batch_size, 101, hidden_units)
            rec_embed = self.item_embed(rec_items.to(self.args.device))
            rec_score = torch.matmul(unified_intention.unsqueeze(1), rec_embed.transpose(2, 1)).squeeze(1)
            return score, rec_score
