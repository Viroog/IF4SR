import torch
import torch.nn as nn
import torch.nn.functional as F


class IF4SR(nn.Module):
    def __init__(self, args, item_num, taxonomy_num):
        super(IF4SR, self).__init__()

        self.item_embed = nn.Embedding(item_num, args.hidden_unit, padding_idx=0)
        # 标签单独初始化
        self.taxonomy_embed = nn.Embedding(taxonomy_num, args.hidden_unit, padding_idx=0)

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

        self.weighted_vector = nn.Linear(args.hidden_unit, 1, bias=False)

        for _ in range(args.gip_block_num):
            # scb
            self.scb_layernorms.append(nn.LayerNorm(args.hidden_unit, eps=1e-8))
            # transform_matrix1: (hidden_unit, L)
            self.scb_transform1.append(nn.Linear(args.L, args.scb_hidden_unit, bias=False))
            # transform_matrix2: (L, hidden_units)
            self.scb_transform2.append(nn.Linear(args.scb_hidden_unit, args.L, bias=False))
            # fcb
            self.fcb_layernorms.append(nn.LayerNorm(args.hidden_unit, eps=1e-8))
            self.fcb_transform1.append(
                nn.Linear(int(args.hidden_unit / args.fcb_head_num), args.fcb_hidden_unit, bias=False))
            self.fcb_transform2.append(
                nn.Linear(args.fcb_hidden_unit, int(args.hidden_unit / args.fcb_head_num), bias=False))
            self.fcb_transform3.append(nn.Linear(args.hidden_unit, args.hidden_unit, bias=False))

        self.args = args
        self.init_params()

    # 初始化参数
    def init_params(self):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
                # 单独初始化
                if name == 'taxonomy_embed.weight':
                    if self.args.taxonomy_init_weight == 'default':
                        pass
                    # 剩余未完成
                    elif self.args.taxonomy_init_weight == 'LINE':
                        pass
                    elif self.args.taxonomy_init_weight == 'word2vec':
                        pass
            except:
                pass

    # 根节点root用于获取最后聚合完的K个意图向量
    # 森林forest本质上是一个非连通图，有K个非联通图。并且每个图都是树形结构
    def get_local_intention(self, root, forest):

        forest = forest.to(self.args.device)
        root = torch.LongTensor(root).to(self.args.device)

        forest.nodes['item'].data['V'] = self.item_embed(forest.nodes['item'].data['id'].to(self.args.device))
        forest.nodes['taxonomy'].data['T'] = self.taxonomy_embed(forest.nodes['taxonomy'].data['id'].to(self.args.device))

        local_intention = None

        return local_intention

    def get_global_intention(self, seq):
        # item embedding: (batch_size, L, hidden_unit) 第0层
        V = self.item_embed(torch.LongTensor(seq).to(self.args.device))

        for i in range(self.args.gip_block_num):
            # scb
            normed_V = self.scb_layernorms[i](V)
            transposed_normed_V = torch.transpose(normed_V, 1, 2)
            scb_out = torch.transpose(
                self.scb_transform2[i](F.gelu(self.scb_transform1[i](transposed_normed_V))), 1, 2)
            V_scb = normed_V + scb_out

            # fcb
            normed_V_scb = self.fcb_layernorms[i](V_scb)
            concated = None
            for h in range(self.args.fcb_head_num):
                # 右边界不包括
                start, end = int(h * (self.args.hidden_unit / self.args.fcb_head_num)), int(
                    (h + 1) * (self.args.hidden_unit / self.args.fcb_head_num))
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

    def forward(self, user, seq, pos, neg, root, forest):

        global_intention = self.get_global_intention(seq)
        local_intention = self.get_local_intention(root, forest)

        pos_embed = self.item_embed(torch.LongTensor(pos).to(self.args.device))
        neg_embed = self.item_embed(torch.LongTensor(neg).to(self.args.device))

        # 计算global intention和local intention的相似度作为权重，进行加权求和

        # 与正负样本预测评分
