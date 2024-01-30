import os
import numpy as np
import torch
    
class MELT(torch.nn.Module):
    def __init__(self, args, logger, train_data, device, item_num, u_L_max, i_L_max, test=False):
        super(MELT, self).__init__()
        self.args = args
        self.logger = logger
        self.device = device
        self.user_branch = USERBRANCH(self.args, device, u_L_max)
        self.item_branch = ITEMBRANCH(self.args, self.device, i_L_max)
        self.train_data = train_data
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, u, seq, pos, neg, u_h_idx, i_h_idx, user_context, item_context, n_item_context, user_thres, item_thres, epoch, sasrec):
        # User branch loss (L_u)
        user_loss = self.user_branch(u_h_idx, sasrec, user_context, user_thres, epoch)
        
        # Item branch loss (L_i)
        item_loss = self.item_branch(i_h_idx, item_context, sasrec, item_thres, self.user_branch.W_U, epoch, n_item_context)
        
        # Next item prediction loss (L_{rec})
        pos_logits, neg_logits = sasrec(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.device), torch.zeros(neg_logits.shape).to(self.device)
        indices = np.where(pos != 0)
        prediction_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        prediction_loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
        
        loss = user_loss * self.args.lamb_u + item_loss * self.args.lamb_i + prediction_loss
        return loss

class USERBRANCH(torch.nn.Module):
    def __init__(self, args, device, u_L_max):
        """
        User branch: Enhance the tail user representation
        """
        super(USERBRANCH, self).__init__()
        self.args = args
        self.W_U = torch.nn.Linear(self.args.hidden_units, self.args.hidden_units)
        torch.nn.init.xavier_normal_(self.W_U.weight.data)
        self.criterion = torch.nn.MSELoss()
        self.device = device
        self.u_L_max = u_L_max
        self.pi = np.pi
              
    def forward(self, u_head_idx, sasrec, user_context, user_thres, epoch):
        full_seq = np.zeros(([len(u_head_idx), self.args.maxlen]), dtype=np.int32)
        w_u_list = []
        
        for i, u_h in enumerate(u_head_idx):
            full_seq[i] = user_context[u_h]
            seq_length = user_context[u_h].nonzero()[0].shape[0]
            
            # Calculate the loss coefficient
            w_u = (self.pi/2)*(epoch/self.args.e_max)+(self.pi/(2*(self.u_L_max-user_thres-1)))*(seq_length-user_thres-1)
            w_u = np.abs(np.sin(w_u))
            w_u_list.append(w_u)
            
        # Representations of full sequence
        full_seq_repre = sasrec.user_representation(full_seq)
        few_seq = np.zeros([len(u_head_idx), self.args.maxlen], dtype=np.int32)
  
        R = np.random.randint(1, user_thres, len(full_seq))
        for i, l in enumerate(R):
            few_seq[i, -l:] = full_seq[i, -l:]
            
        # Representations of recent interactions 
        few_seq_repre = sasrec.user_representation(few_seq)
        w_u_list = torch.FloatTensor(w_u_list).view(-1,1).to(self.device)
        
        # Curriculum Learning by user
        loss = (w_u_list*((self.W_U(few_seq_repre) - full_seq_repre) ** 2)).mean()
        return loss

class ITEMBRANCH(torch.nn.Module):
    def __init__(self, args, device, i_L_max):
        """
        Item branch: Enhance the tail item representation
        """
        super(ITEMBRANCH, self).__init__()
        self.args = args
        self.device = device
        self.W_I = torch.nn.Linear(args.hidden_units, args.hidden_units)
        torch.nn.init.xavier_normal_(self.W_I.weight.data)
        self.criterion = torch.nn.MSELoss()
        self.i_L_max = i_L_max
        self.pi = np.pi
          
    def forward(self, i_head_idx, item_context, sasrec, item_thres, W_U, epoch=None, n_item_context=None):
        target_embed = []
        subseq_set = []
        subseq_set_idx = [0]
        idx = 0
        w_i_list = []
        
        for i, h_i in enumerate(i_head_idx):
            item_context_list = np.vstack(item_context[h_i])
            n_context = min(self.i_L_max, n_item_context[h_i])
            
            #  Calculate the loss coefficient
            w_i = (self.pi/2)*(epoch/self.args.e_max)+(self.pi/100)*(n_context-(item_thres+1))
            w_i = np.abs(np.sin(w_i))
            w_i_list.append(w_i)
            len_context = len(item_context[h_i])
            
            # Set upper bound of item freq.
            thres = min(len_context, item_thres) 
            n_few_inter = np.random.randint(1, thres+1)
            
            # Randomly sample the contexts
            K = np.random.choice(range(len(item_context_list)), int(n_few_inter), replace=False)
            idx += len(K)
            
            subseq_set.append(item_context_list[K])
            target_embed.append(h_i)
            subseq_set_idx.append(idx)
        
        # Encode the subsequence set
        subseq_repre_set = sasrec.user_representation(np.vstack(subseq_set))
        
        # Knowledge transfer from user to item branch
        subseq_repre_set = subseq_repre_set + W_U(subseq_repre_set) 
        
        # Contextualized representations
        subseq_set = []
        for i, h_i in enumerate(i_head_idx):
            mean_context = subseq_repre_set[subseq_set_idx[i]:subseq_set_idx[i+1]]
            subseq_set.append(mean_context.mean(0))
        
        few_subseq_embed = torch.stack(subseq_set)
        target_embed = torch.LongTensor(target_embed).to(self.device)
        w_i_list = torch.FloatTensor(w_i_list).view(-1,1).to(self.device)
        
        # Curriculum Learning by item
        loss = (w_i_list*((self.W_I(few_subseq_embed) - sasrec.item_emb(target_embed))) ** 2).mean()
        return loss

class SASRec(torch.nn.Module):
    """
    Parameter of SASRec
    """
    def __init__(self, args, item_num):
        super(SASRec, self).__init__()
        self.args = args
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"

        self.item_emb = torch.nn.Embedding(item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """
        Sequence Encoder: f_{\theta}(S_u)
        """
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                         
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices, u_tail, u_transfer = None):
        """
        MELT - Prediction
        """
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]
        if u_transfer:
            # Knowledge transfer to tail users
            final_feat[u_tail] = u_transfer(final_feat[u_tail]) + final_feat[u_tail] 

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device)) 

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def update_tail_item_representation(self, i_tail_loader, item_context, W_I):
        """
        Update all the tail item embeddings
        """        
        for _, i_t in enumerate(i_tail_loader):
            tail_idx = []
            collect_context = []
            i_tail_idx = [0]
            cumulative_idx = 0
            for i in i_t[0]:
                i = i.item()
                if len(item_context[i]) >=1:
                    stack_context = np.vstack(item_context[i])
                    cumulative_idx += len(stack_context)
                    i_tail_idx.append(cumulative_idx)
                    
                    collect_context.extend(stack_context)
                    tail_idx.append(i)
            group_context_embed = self.user_representation(np.vstack(collect_context))
            i_tail_average_emb = []
            idx = 0 
            for i in i_t[0]:
                i = i.item()
                if len(item_context[i]) >=1:
                    i_encode_emb = group_context_embed[i_tail_idx[idx]:i_tail_idx[idx+1]]
                    i_tail_average_emb.append(i_encode_emb.mean(0))
                    idx+=1
            group_fully_context_embed = torch.stack(i_tail_average_emb)
            i_tail_estimate_embed = W_I(group_fully_context_embed) 
            
            tail_idx = torch.LongTensor(tail_idx).to(self.device)
            self.item_emb.weight.data[tail_idx] = i_tail_estimate_embed # Direclty update the tail item embedding
    
    def user_representation(self, log_seqs):
        """
        User representation
        """
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        return final_feat
    
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs