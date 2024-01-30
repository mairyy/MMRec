import os
import torch
import numpy as np
from params import parse_args
import torch.utils.data as data_utils
from embedder import embedder
from utils import setupt_logger, set_random_seeds, Checker
from sampler import NegativeSampler
from data import ValidData, TestData, TrainData
from models import MELT, SASRec

class Trainer(embedder):
    def __init__(self, args):
        self.logger = setupt_logger(args, f'log/{args.model}/{args.dataset}', name = args.model, filename = "log.txt")
        embedder.__init__(self, args, self.logger)
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"
        self.args = args
        torch.cuda.set_device(self.device)
        self.split_head_tail()
        self.save_user_item_context()
        
    def train(self):
        set_random_seeds(self.args.seed)
        u_L_max = self.args.maxlen
        i_L_max = self.item_threshold + 50
        self.sasrec = SASRec(self.args, self.item_num).to(self.device)
        self.melt = MELT(self.args, self.logger, self.train_data, self.device, self.item_num, u_L_max, i_L_max).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        
        # Build the train, valid, test loader
        train_dataset = TrainData(self.train_data, self.user_num, self.item_num, batch_size=self.args.batch_size, maxlen=self.args.maxlen)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        valid_dataset = ValidData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.valid_loader = data_utils.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        # For selecting the best model
        self.validcheck = Checker(self.logger)

        adam_optimizer = torch.optim.Adam([{"params": self.melt.parameters()}, {"params": self.sasrec.parameters()}], lr=self.args.lr, betas=(0.9, 0.98))
        
        i_tail_set = data_utils.TensorDataset(torch.LongTensor(list(self.i_tail_set)))
        
        # For updating the tail item embedding
        i_tail_loader = data_utils.DataLoader(i_tail_set, self.args.batch_size * 2, shuffle=False, drop_last=False) # No matter how batch size is
        
        # DataLoader for bilateral branches
        i_batch_size = len(self.i_head_set) // (len(train_loader)-1) + 1
        i_h_loader = data_utils.DataLoader(self.i_head_set, i_batch_size, shuffle=True, drop_last=False)
        u_batch_size = len(self.u_head_set) // (len(train_loader)-1) + 1
        u_h_loader = data_utils.DataLoader(self.u_head_set, u_batch_size, shuffle=True, drop_last=False)
        
        for epoch in range(self.args.e_max):
            self.sasrec.train()
            training_loss = 0.0
            self.melt.eval()
            
            with torch.no_grad():
                # Knowledge transfer from item branch to user branch
                self.sasrec.update_tail_item_representation(i_tail_loader, self.item_context, self.melt.item_branch.W_I) 
                #i_tail_loader, item_context, W_I
                
            self.melt.train()
            for _, ((u,seq,pos,neg),(u_idx), (i_idx)) in enumerate(zip(train_loader, u_h_loader, i_h_loader)): #pos: matrix contain seq[1->end], neg:
                adam_optimizer.zero_grad()
                u = np.array(u); seq = np.array(seq); pos = np.array(pos); neg = np.array(neg)
                u_idx = u_idx.numpy()
                i_idx = i_idx.numpy()

                loss = self.melt(u, seq, pos, neg, u_idx, i_idx, self.user_context, self.item_context, self.n_item_context, \
                                  self.user_threshold, self.item_threshold, epoch, self.sasrec)
                loss.backward()
                adam_optimizer.step()
                training_loss += loss.item()

            if epoch % 2 == 0:
                with torch.no_grad():
                    self.sasrec.eval()
                    self.melt.eval()
                    result_valid = self.evaluate(self.melt, self.sasrec, k=10, is_valid='valid')
                    best_valid = self.validcheck(result_valid, epoch, self.melt, f'{self.args.model}_{self.args.dataset}.pth')
                    self.validcheck.print_epoch(result_valid, epoch, training_loss, self.args)
        
            # if epoch % 10 == 0:
            #     print(f'Epoch: {epoch}, Evaluating: Dataset({self.args.dataset}), Loss: ({training_loss:.2f}), GPU: {self.args.gpu}')
                
        # Evaluation
        with torch.no_grad():
            self.sasrec.eval()
            self.validcheck.best_model.eval()
            result_5 = self.evaluate(self.validcheck.best_model, self.sasrec, k=5, is_valid='test')
            result_10 = self.evaluate(self.validcheck.best_model, self.sasrec, k=10, is_valid='test')
            result_20 = self.evaluate(self.validcheck.best_model, self.sasrec, k=20, is_valid='test')
            result_50 = self.evaluate(self.validcheck.best_model, self.sasrec, k=50, is_valid='test')
            self.validcheck.refine_test_result(result_5, result_10, result_20, result_50)
            self.validcheck.print_result()
   
        # folder = f"save_model/{self.args.dataset}"
        # os.makedirs(folder, exist_ok=True)
        # torch.save(self.validcheck.best_model.state_dict(), os.path.join(folder, self.validcheck.best_name))
        self.save_model()
        self.validcheck.print_result()


    def test(self):
        set_random_seeds(self.args.seed)
        user_max_thres = self.args.maxlen
        item_max_thres = self.item_threshold + self.args.maxlen
        self.sasrec = SASRec(self.args, self.item_num).to(self.device)
        self.melt = MELT(self.args, self.logger, self.train_data, self.device, self.item_num, user_max_thres, item_max_thres, test=True).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)

        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        self.printer = Checker(self.logger)

        os.makedirs(f"save_model/{self.args.dataset}", exist_ok=True)

        # model_path = f"save_model/{self.args.dataset}/{self.args.model}_{self.args.dataset}.pth"
        # self.melt.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.load_model()
        self.sasrec.eval()
        self.melt.eval()

        # Evaluate
        result_5 = self.evaluate(self.melt, self.sasrec, k=5, is_valid='test')
        result_10 = self.evaluate(self.melt, self.sasrec, k=10, is_valid='test')
        result_20 = self.evaluate(self.melt, self.sasrec, k=20, is_valid='test')
        result_50 = self.evaluate(self.melt, self.sasrec, k=50, is_valid='test')
        self.printer.refine_test_result(result_5, result_10, result_20, result_50)
        self.printer.print_result()


    def evaluate(self, melt, sasrec, k, is_valid='test'):
        """
        Evaluation on validation or test set
        """
        HIT = 0.0  # Overall Hit
        NDCG = 0.0 # Overall NDCG
        
        TAIL_USER_NDCG = 0.0
        HEAD_USER_NDCG = 0.0
        TAIL_ITEM_NDCG = 0.0
        HEAD_ITEM_NDCG = 0.0
        
        TAIL_USER_HIT = 0.0
        HEAD_USER_HIT = 0.0
        TAIL_ITEM_HIT = 0.0
        HEAD_ITEM_HIT = 0.0
        
        n_all_user = 0.0
        n_head_user = 0.0
        n_tail_user = 0.0
        n_head_item = 0.0
        n_tail_item = 0.0
        
        loader = self.test_loader if is_valid == 'test' else self.valid_loader
        
        for _, (u, seq, item_idx, test_idx) in enumerate(loader):
            u_head = (self.u_head_set[None, ...] == u.numpy()[...,None]).nonzero()[0]         # Index of head users
            u_tail = np.setdiff1d(np.arange(len(u)), u_head)                                  # Index of tail users
            i_head = (self.i_head_set[None, ...] == test_idx.numpy()[...,None]).nonzero()[0]  # Index of head items
            i_tail = np.setdiff1d(np.arange(len(u)), i_head)                                  # Index of tail items
            
            predictions = -sasrec.predict(seq.numpy(), item_idx.numpy(), u_tail, melt.user_branch.W_U) # Sequence Encoder
            
            rank = predictions.argsort(1).argsort(1)[:,0].cpu().numpy()
            n_all_user += len(predictions)
            hit_user = rank < k
            ndcg = 1 / np.log2(rank + 2)

            n_head_user += len(u_head)
            n_tail_user += len(u_tail)
            n_head_item += len(i_head)
            n_tail_item += len(i_tail)
            
            HIT += np.sum(hit_user).item()
            HEAD_USER_HIT += sum(hit_user[u_head])
            TAIL_USER_HIT += sum(hit_user[u_tail])
            HEAD_ITEM_HIT += sum(hit_user[i_head])
            TAIL_ITEM_HIT += sum(hit_user[i_tail])
            
            NDCG += np.sum(1 / np.log2(rank[hit_user] + 2)).item()
            HEAD_ITEM_NDCG += sum(ndcg[i_head[hit_user[i_head]]])
            TAIL_ITEM_NDCG += sum(ndcg[i_tail[hit_user[i_tail]]])
            HEAD_USER_NDCG += sum(ndcg[u_head[hit_user[u_head]]])
            TAIL_USER_NDCG += sum(ndcg[u_tail[hit_user[u_tail]]])
            
        result = {'Overall': {'NDCG': NDCG / n_all_user, 'HIT': HIT / n_all_user}, 
                'Head_User': {'NDCG': HEAD_USER_NDCG / n_head_user, 'HIT': HEAD_USER_HIT / n_head_user},
                'Tail_User': {'NDCG': TAIL_USER_NDCG / n_tail_user, 'HIT': TAIL_USER_HIT / n_tail_user},
                'Head_Item': {'NDCG': HEAD_ITEM_NDCG / n_head_item, 'HIT': HEAD_ITEM_HIT / n_head_item},
                'Tail_Item': {'NDCG': TAIL_ITEM_NDCG / n_tail_item, 'HIT': TAIL_ITEM_HIT / n_tail_item}
                }

        return result

    def save_model(self):
        folder = f"save_model/{self.args.dataset}"
        os.makedirs(folder, exist_ok=True)

        content = {
            'sasrec': self.sasrec.state_dict(),
            'melt': self.validcheck.best_model.state_dict(),
        }

        torch.save(content, os.path.join(folder, self.validcheck.best_name))

    def load_model(self):
        folder = f"save_model/{self.args.dataset}/{self.args.model}_{self.args.dataset}.pth"
        ckp = torch.load(folder)
        self.sasrec.load_state_dict(ckp['sasrec'])
        self.melt.load_state_dict(ckp['melt'])

def main():
    args = parse_args() 
    torch.set_num_threads(4)

    if args.model == "MELT":
        embedder = Trainer(args)
    
    if args.inference:
        embedder.test()
    else:
        embedder.train()

if __name__ == "__main__":
    main()
