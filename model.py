import torch.nn.functional as F
import pickle
from statistics import mean
from params import args
from torch import nn
import numpy as np
import torch as t
import math
import time
from transformer import *

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform

class Item_Graph(nn.Module):
    def __init__(self, dataset, handler):
        super(Item_Graph, self).__init__()
        self.knn_k = 5
        self.k = 40
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.mm_image_weight = 0.2
        # self.mode = args.mode
        # self.att_mode = args.att_mode
        
        #if attention mode
        self.t_weight = nn.Parameter(init(t.empty(args.f_dim, int(args.latdim/2)))).to(self.device)
        self.t_bias = nn.Parameter(init(t.empty(args.item, int(args.latdim/2)))).to(self.device)
        self.v_weight = nn.Parameter(init(t.empty(args.f_dim, int(args.latdim/2)))).to(self.device)
        self.v_bias = nn.Parameter(init(t.empty(args.item, int(args.latdim/2)))).to(self.device)

        self.text_embedding = nn.Embedding.from_pretrained(handler.t_feat, freeze=False).to(self.device)
        self.image_embedding = nn.Embedding.from_pretrained(handler.v_feat, freeze=False).to(self.device)

        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(args.num_gcn_layers)])

        self.t_feat = t.matmul(handler.t_feat.to(self.device), self.t_weight) + self.t_bias
        self.v_feat = t.matmul(handler.v_feat.to(self.device), self.v_weight) + self.v_bias
        self.item_rep = t.cat((self.v_feat, self.t_feat), dim=1)
        
    # if self.v_feat is not None:
    # if (self.mode == 'graph_v'):
        indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        self.mm_adj = image_adj
    # if self.t_feat is not None:
    # if (self.mode == 'graph_t'):
        indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.mm_adj = text_adj
    # if self.v_feat is not None and self.t_feat is not None:
    # if (self.mode == 'graph_t_v' or self.mode == 'remove'):
        self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj

    def forward(self):
        # if (self.mode == 'graph_t_v'):
        item_rep = self.item_rep
        # elif (self.mode == 'graph_t'):
        #     item_rep = self.t_feat
        # elif (self.mode == 'text'):
        #     return t.matmul(self.t_feat, self.weight) + self.bias
        # elif (self.mode == 't_v'):
        #     item_rep = t.cat((self.v_feat, self.t_feat), dim=1)
        #     return t.matmul(item_rep, self.weight) + self.bias
        # elif (self.mode == 'attention_v_t'):
        #     t_feat = t.matmul(self.t_feat, self.t_weight) + self.t_bias
        #     v_feat = t.matmul(self.v_feat, self.v_weight) + self.v_bias
        #     return t_feat, v_feat

        # item_rep = t.matmul(item_rep, self.weight) + self.bias
        h = item_rep
        for i in self.gcn_layers:
            h = t.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h
        return item_rep

    def get_knn_adj_mat(self, mm_embedding):
        context_norm = mm_embedding.div(t.norm(mm_embedding, p=2, dim=-1, keepdim=True))
        sim = t.mm(context_norm, context_norm.transpose(1, 0))
        #k = 5
        _, knn_ind = t.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = t.arange(knn_ind.shape[0]).to(self.device)
        indices0 = t.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = t.stack((t.flatten(indices0), t.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = t.sparse.FloatTensor(indices, t.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + t.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = t.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return t.sparse.FloatTensor(indices, values, adj_size)
    
    
def sparse_dropout(x, keep_prob):
    msk = (t.rand(x._values().size()) + keep_prob).floor().type(t.bool)
    idx = x._indices()[:, msk]
    val = x._values()[msk]
    return t.sparse.FloatTensor(idx, val, x.shape).cuda()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.layers = nn.Sequential(
            nn.Linear(512, 256).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(256, 32).to(self.device)
        )  
    def forward(self, x):
        x = self.layers(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.item_id = init(t.empty(args.item, args.latdim)) # args.item = num_real_item + 1
        self.item_emb = nn.Parameter(self.item_id) 
        self.item_id = self.item_id.to(self.device)
        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(args.num_gcn_layers)])

    def get_ego_embeds(self):
        return self.item_emb

    def forward(self, encoder_adj):
        embeds = [self.item_emb]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds

class TrivialDecoder(nn.Module):
    def __init__(self):
        super(TrivialDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.latdim * 3, args.latdim, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        pos_emb.append(embeds[-1][pos[:,0]])
        pos_emb.append(embeds[-1][pos[:,1]])
        pos_emb.append(embeds[-1][pos[:,0]] * embeds[-1][pos[:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]])
        neg_emb.append(embeds[-1][neg[:,:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]] * embeds[-1][neg[:,:,1]])
        pos_emb = t.cat(pos_emb, -1) # (n, latdim * 3)
        neg_emb = t.cat(neg_emb, -1) # (n, num_reco_neg, latdim * 3)
        pos_scr = t.exp(t.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = t.exp(t.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = t.sum(neg_scr, -1) + pos_scr
        loss = -t.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.latdim * args.num_gcn_layers ** 2, args.latdim * args.num_gcn_layers, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim * args.num_gcn_layers, args.latdim, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        for i in range(args.num_gcn_layers):
            for j in range(args.num_gcn_layers):
                pos_emb.append(embeds[i][pos[:,0]] * embeds[j][pos[:,1]])
                neg_emb.append(embeds[i][neg[:,:,0]] * embeds[j][neg[:,:,1]])
        pos_emb = t.cat(pos_emb, -1) # (n, latdim * num_gcn_layers ** 2)
        neg_emb = t.cat(neg_emb, -1) # (n, num_reco_neg, latdim * num_gcn_layers ** 2)
        pos_scr = t.exp(t.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = t.exp(t.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = t.sum(neg_scr, -1) + pos_scr
        loss = -t.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

class SASRec(nn.Module):
    def __init__(self, handler):
        super(SASRec, self).__init__()
        self.handler = handler
        self.device = args.device
        self.pos_emb = nn.Parameter(init(t.empty(args.max_seq_len, args.latdim)))
        self.layers = nn.Sequential(*[TransformerLayer() for i in range(args.num_trm_layers)])
        self.LayerNorm = nn.LayerNorm(args.latdim)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.apply(self.init_weights)
        self.attention = DIFTransformerEncoder()
        self.t_weight = nn.Parameter(init(t.empty(args.f_dim, args.latdim))).to(self.device)
        self.t_bias = nn.Parameter(init(t.empty(args.item, args.latdim))).to(self.device)
        self.v_weight = nn.Parameter(init(t.empty(args.f_dim, args.latdim))).to(self.device)
        self.v_bias = nn.Parameter(init(t.empty(args.item, args.latdim))).to(self.device)

        self.t_feat = t.matmul(self.handler.t_feat.to(self.device), self.t_weight) + self.t_bias
        self.v_feat = t.matmul(self.handler.v_feat.to(self.device), self.v_weight) + self.v_bias
        # if (args.mode == 'attention_v_t'):
        #     self.t_feat, self.v_feat = Item_Graph(dataset=args.data).forward()
        #     # print(t_feat.shape, v_feat.shape)
        #     # self.t_feat = t_feat.unsqueeze(1)
        #     # self.v_feat = v_feat.unsqueeze(1)
        #     # print(self.t_feat.shape, self.v_feat.shape)
        #     # self.all_feat = t.stack([t_feat, v_feat], dim=0)
        #     self.attention = DIFTransformerEncoder()
        # else:
        #     self.item_rep = Item_Graph(dataset=args.data)
        #     self.item_emb = self.item_rep().cuda()
    
    def get_seq_emb(self, sequence, item_emb, mm_emb):
        # print('1', sequence, sequence.size(0))
        seq_len = sequence.size(1)
        # print('2', seq_len)
        pos_ids = t.arange(seq_len, dtype=t.long, device=sequence.device)
        # print('3', pos_ids)
        pos_ids = pos_ids.unsqueeze(0).expand_as(sequence)
        # print('4', pos_ids)
        # print('5', item_emb, item_emb.shape)
        itm_emb = item_emb[sequence]
        # print('5\'', itm_emb, itm_emb.shape)
        pos_emb = self.pos_emb[pos_ids]
        if (args.mode == 'remove'):
            extended_attention_mask = self.get_attention_mask(sequence)
            t_feat = self.t_feat[sequence]
            v_feat = self.v_feat[sequence]
            mm_emb = mm_emb[sequence]
            # t_feat = t_feat.unsqueeze(1)
            # v_feat = v_feat.unsqueeze(1)
            # print(t_feat.shape, v_feat.shape)
            # print(self.all_feat.shape)
            # all_feat = t.stack([t_feat, v_feat], dim=0)
            all_feat = []
            all_feat.append(t_feat)
            all_feat.append(v_feat)
            all_feat.append(mm_emb)
            # all_feat = t.cat((t_feat, v_feat))
            # all_feat = t_feat
            # print('1', all_feat.shape)
            seq_emb_list = self.attention(itm_emb, all_feat, pos_emb, extended_attention_mask, True)
            # print(seq_emb_list)
            # seq_emb = seq_emb_list
            # seq_emb = []
            # for seq in seq_emb_list:
            #     seq = self.LayerNorm(seq)
            #     seq = self.dropout(seq)
            #     seq_emb.append(seq)
            seq_emb = seq_emb_list[-1]
            seq_emb = seq_emb + pos_emb
        else:
            seq_emb = itm_emb + pos_emb
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        return seq_emb

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = t.triu(t.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, item_emb, mm_emb):
        if (args.att_mode == 'remove'):
            #remove transformer
            seq_emb = self.get_seq_emb(input_ids, item_emb, mm_emb)
            # sequence = input_ids
            # seq_len = sequence.size(1)
            # pos_ids = t.arange(seq_len, dtype=t.long, device=sequence.device)
            # pos_ids = pos_ids.unsqueeze(0).expand_as(sequence)
            # itm_emb = item_emb[sequence]
            # pos_emb = self.pos_emb[pos_ids]
            # extended_attention_mask = self.get_attention_mask(sequence)
            # t_feat = self.t_feat[sequence]
            # v_feat = self.v_feat[sequence]
            # all_feat = []
            # all_feat.append(t_feat)
            # all_feat.append(v_feat)
            # seq_emb_list = self.attention(itm_emb, all_feat, pos_emb, extended_attention_mask, True)
            # seq_emb = seq_emb_list[-1]
            # seq_emb = self.LayerNorm(seq_emb)
            # seq_emb = self.dropout(seq_emb)
        
        else:
        #no remove transformer
            attention_mask = (input_ids > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)

            subsequent_mask = t.triu(t.ones(attn_shape), diagonal=1) # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long()
            subsequent_mask = subsequent_mask.cuda()

            extended_attention_mask = extended_attention_mask * subsequent_mask
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            seq_embs = [self.get_seq_emb(input_ids, item_emb, mm_emb)]
            for trm in self.layers:
                seq_embs.append(trm(seq_embs[-1], extended_attention_mask))
            seq_emb = sum(seq_embs)

        return seq_emb

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class TransformerLayer(nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttentionLayer()
        self.intermediate = IntermediateLayer()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output

class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.latdim / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.latdim, self.all_head_size)
        self.key = nn.Linear(args.latdim, self.all_head_size)
        self.value = nn.Linear(args.latdim, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.latdim, args.latdim)
        self.LayerNorm = nn.LayerNorm(args.latdim)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.apply(self.init_weights)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = t.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = t.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class IntermediateLayer(nn.Module):
    def __init__(self):
        super(IntermediateLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.latdim, args.latdim * 4, bias=True),
            nn.GELU(),
            nn.Linear(args.latdim * 4, args.latdim, bias=True),
            nn.Dropout(args.hidden_dropout_prob),
            nn.LayerNorm(args.latdim)
        )

    def forward(self, x):
        return self.layers(x)

class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()

    def make_noise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise

    def forward(self, adj, embeds, foo=None):
        order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1]) #convert adj into a column vector 
        fstEmbeds = t.spmm(adj, embeds) - embeds
        fstNum = order

        emb = [fstEmbeds]
        num = [fstNum]

        for i in range(args.mask_depth):
            adj = sparse_dropout(adj, args.path_prob ** (i + 1))
            emb.append((t.spmm(adj, emb[-1]) - emb[-1]) - order * emb[-1])
            num.append((t.spmm(adj, num[-1]) - num[-1]) - order)
            order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])

        subgraphEmbeds = sum(emb) / (sum(num) + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)

        embeds = F.normalize(embeds, p=2)
        scores = t.sum(subgraphEmbeds * embeds, dim=-1)
        scores = self.make_noise(scores)

        _, candidates = t.topk(scores, args.num_mask_cand)  #return indices of top k item

        return scores, candidates

class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()

    def normalize(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        masked_rows = []
        masked_cols = []
        masked_idct = []

        for i in range(args.mask_depth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            idct = None
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                if idct == None:
                    idct = t.logical_or(rowIdct, colIdct)
                else:
                    idct = t.logical_or(idct, t.logical_or(rowIdct, colIdct))
            nxtRows = rows[idct]
            nxtCols = cols[idct]
            masked_rows.extend(nxtRows)
            masked_cols.extend(nxtCols)
            rows = rows[t.logical_not(idct)]
            cols = cols[t.logical_not(idct)]
            nxtSeeds = nxtRows + nxtCols
            if len(nxtSeeds) > 0 and i != args.mask_depth - 1:
                nxtSeeds = t.unique(nxtSeeds)
                cand = t.randperm(nxtSeeds.shape[0])
                nxtSeeds = nxtSeeds[cand[:int(nxtSeeds.shape[0] * args.path_prob ** (i + 1))]] # the dropped edges from P^k

        masked_rows = t.unsqueeze(t.LongTensor(masked_rows), -1) 
        masked_cols = t.unsqueeze(t.LongTensor(masked_cols), -1)
        masked_edge = t.hstack([masked_rows, masked_cols])
        encoder_adj = self.normalize(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

        return encoder_adj, masked_edge
