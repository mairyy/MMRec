from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import pickle

class embedder(torch.nn.Module):
    def __init__(self, args, logger):
        """
        For generating train, valid, test data
        """
        super(embedder, self).__init__()
        self.args = args
        self.logger = logger
        self.dataset = self.data_partition(args.dataset)
        [self.train_data, self.valid_data, self.test_data, self.user_num, self.item_num] = self.dataset
        self.num_batch = len(self.train_data) // args.batch_size

    def split_head_tail(self):
        """
        Split user and items into head or tail sets
        """
        all_seq_len = []
        all_item_train = []
        u_head_set = set()
        u_tail_set = set()

        for k, v in self.train_data.items():
            all_seq_len.append(len(v))
            all_item_train.extend(v)

        sort_seq_len = sorted(all_seq_len, reverse=False)
        user_threshold = np.quantile(sort_seq_len, self.args.pareto_rule)

        for k, v in self.train_data.items():
            if len(v) <= user_threshold:
                u_tail_set.add(k)
            else:
                u_head_set.add(k)

        item_count = Counter(all_item_train)
        item_train_set = np.array(list(set(all_item_train)))
        sort_item_freq = dict(sorted(item_count.items(), key=lambda d:d[1], reverse=True))
        
        self.item_threshold = np.quantile(list(sort_item_freq.values()), self.args.pareto_rule - 0.02)
        self.logger.info(f"User Thres: {user_threshold}, Item Thres: {self.item_threshold}, Pareto({self.args.pareto_rule})")
        i_head_set = set()
        i_tail_set = set()

        for i in range(1, self.item_num+1):
            if i not in item_train_set:
                i_tail_set.add(i)
            else:
                if item_count[i] <= self.item_threshold:
                    i_tail_set.add(i)
                else:
                    i_head_set.add(i)

        self.user_threshold = user_threshold
        self.u_head_set = np.array(list(u_head_set))
        self.i_head_set = np.array(list(i_head_set))
        self.i_tail_set = np.array(list(i_tail_set))
        self.u_tail_set = u_tail_set
        self.item_train_set = item_train_set
        self.item_count = item_count

    
    def save_user_item_context(self):
        """
        Preprocess the item interaction set (i.e., C_i)
        """
        user_context = {}
        item_context = defaultdict(list)
        n_item_context = defaultdict(int)
        
        for k, seq in self.train_data.items():
            for i, item  in enumerate(seq):
                
                if i ==0: i = 0       
                all_seq = np.zeros(self.args.maxlen, dtype=np.int32)
                idx = self.args.maxlen - 1
                for j in reversed(seq[:i+1]):
                    all_seq[idx] = j
                    idx -= 1
                    if idx == -1: break
                item_context[item].append(all_seq)
                n_item_context[item] += 1
                if len(seq) > 1:
                    if i == len(seq) -1 : i = len(seq)-1
                    all_seq = np.zeros(self.args.maxlen, dtype=np.int32)
                    idx = self.args.maxlen - 1
                    for j in seq[i:]:
                        all_seq[idx] = j
                        idx -= 1
                        if idx == -1: break
                    item_context[item].append(all_seq)
                    
            all_seq = np.zeros(self.args.maxlen, dtype=np.int32)
            idx = self.args.maxlen - 1
            for j in reversed(seq):
                all_seq[idx] = j
                idx -= 1
                if idx == -1: break
            user_context[k] = all_seq

        self.n_item_context = n_item_context
        self.item_context = item_context
        self.user_context = user_context
           

    def data_partition(self, fname):
        
        """
        Bring the train, valid, test data
        """
        num = {'usernum':0, 'itemnum':0, 'max_user_length': 0, 'min_user_length': 100000, 'test_user': 0, 'total_interaction':0}
        user_train = {}
        user_valid = {}
        user_test = {}
        total_item = defaultdict(int)
        with open('datasets/%s/tst' % fname, 'rb') as out:
            data = pickle.load(out)
        
        #split train, valid, test
        for i, user in enumerate(data):
            nfeedback = len(user)
            num['total_interaction'] += nfeedback
            total_item_feedback = user

            num['usernum'] = max(i+1, num['usernum'])
            num['itemnum'] = max(num['itemnum'], max(total_item_feedback))

            if nfeedback < 3:
                user_train[i+1] = total_item_feedback
                user_valid[i+1] = []
                user_test[i+1] = []
            else:
                num['test_user'] += 1
                user_train[i+1] = total_item_feedback[:-2]
                user_valid[i+1] = []
                user_valid[i+1].append(total_item_feedback[-2])
                user_test[i+1] = []
                user_test[i+1].append(total_item_feedback[-1])

            if num['max_user_length'] < len(total_item_feedback):
                num['max_user_length'] = len(total_item_feedback)
            if num['min_user_length'] > len(total_item_feedback):
                num['min_user_length'] = len(total_item_feedback)

            for j in total_item_feedback:
                total_item[j] +=1

        min_item_length = min(total_item.values())
        max_item_length = max(total_item.values())
        self.logger.info(f"Max user seq : {num['max_user_length']}, Max item freq : {max_item_length}, Min item freq : {min_item_length}, Min user seq : {num['min_user_length']}")
        self.logger.info(f"Test User(>=3) : {num['test_user']}")
        self.logger.info(f"Total User Number : {num['usernum']}, Item Number : {num['itemnum']}")
        self.logger.info(f"Total Interaction : {num['total_interaction']}")
        
        total_length = 0.0
        for u in user_train:
            total_length += len(user_train[u])
            
        self.logger.info(f"Average Sequence Length: {total_length/len(user_train):.2f}") # Average of sequence in train data
        return [user_train, user_valid, user_test, num['usernum'], num['itemnum']]