import logger as logger
from params import args
from logger import log
from handler import *
from model import *
from utils import *
import numpy as np
import torch as t
import pickle
import sys
import os
import csv

t.manual_seed(args.seed)
np.random.seed(args.seed)

class Coach:
    def __init__(self, handler):
        self.handler = handler

        log(f"Users: {args.user}, Items(+1): {args.item}")
        self.metrics = dict()
        mets = ['loss', 'loss_main', 'hr@5', 'ndcg@5', 'hr@10', 'ndcg@10', 'hr@20', 'ndcg@20', 'hr@50', 'ndcg@50']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def make_print(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '                   '
        return ret

    def run(self):
        self.prepare_model()
        log('Model Prepared')
        if args.load_model != None:
            self.load_model()
            stloc = len(self.metrics['Trainloss']) * args.test_frequency - (args.test_frequency - 1)
        else:
            stloc = 0
            log('Model Initialized')
        bestRes = None
        reses = self.test_epoch()
        for ep in range(stloc, args.epoch):
            tst_flag = (ep % args.test_frequency == 0)
            reses = self.train_epoch()
            log(self.make_print('Train', ep, reses, tst_flag))
            sys.stdout.flush()
            if tst_flag:
                reses = self.test_epoch()
                log(self.make_print('Test', ep, reses, tst_flag))
                sys.stdout.flush()
                if bestRes is None or reses['hr@10'] > bestRes['hr@10']:
                    bestRes = reses
                    log(self.make_print('Best Result', args.epoch, bestRes, True), bold=True)
                    self.save_history()
            print()
        reses = self.test_epoch(True)
        log(self.make_print('Test', args.epoch, reses, True))
        log(self.make_print('Best Result', args.epoch, bestRes, True), bold=True)

    def prepare_model(self):
        self.encoder = Encoder().cuda()
        self.recommender = SASRec().cuda()
        self.opt = t.optim.Adam(
            [{"params": self.encoder.parameters()},
             {"params": self.recommender.parameters()}],
            lr=args.lr, weight_decay=0
        )


    def train_epoch(self):
        # self.encoder.train()
        self.recommender.train()

        loss_his = []
        ep_loss, ep_loss_main = 0, 0, 
        trn_loader = self.handler.trn_loader
        steps = trn_loader.dataset.__len__() // args.trn_batch

        for i, batch_data in enumerate(trn_loader):

            batch_data = [i.cuda() for i in batch_data]
            seq, pos, neg = batch_data

            item_emb, item_emb_his = self.encoder(self.handler.ii_adj)
            seq_emb = self.recommender(seq, item_emb)
            tar_msk = pos > 0
            loss_main = cross_entropy(seq_emb, item_emb[pos], item_emb[neg], tar_msk)

            loss_regu = (calc_reg_loss(self.encoder) + calc_reg_loss(self.recommender)) * args.reg

            loss = loss_main + loss_regu
            loss_his.append(loss_main)

            ep_loss += loss.item()
            ep_loss_main += loss_main.item()
            log('Step %d/%d: loss = %.3f, loss_main = %.3f loss_regu = %.3f' % (i, steps, loss, loss_main, loss_regu), save=False, oneline=True)
            sys.stdout.flush()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        ret = dict()
        ret['loss'] = ep_loss / steps
        ret['loss_main'] = ep_loss_main / steps

        return ret

    def test_epoch(self, logPredict=False):
        self.encoder.eval()
        self.recommender.eval()
        tst_loader = self.handler.tst_loader
        ep_h5, ep_n5, ep_h10, ep_n10, ep_h20, ep_n20, ep_h50, ep_n50  = [0] * 8
        group_h20 = [0] * 4
        group_n20 = [0] * 4
        group_num = [0] * 4
        num = tst_loader.dataset.__len__()
        steps = num // args.tst_batch
        item_emb, item_emb_his = self.encoder(self.handler.ii_adj)


        with t.no_grad():
            for i, batch_data in enumerate(tst_loader):
                batch_data = [i.cuda() for i in batch_data]
                seq, pos, neg = batch_data
                seq_emb = self.recommender(seq, item_emb)
                seq_emb = seq_emb[:,-1,:] # (batch, 1, latdim)
                all_ids = t.arange(1, item_emb.shape[0]).unsqueeze(0).expand(seq.shape[0], -1).cuda() # (batch, 100)
                all_emb = item_emb[all_ids] # (batch, 100, latdim)
                all_scr = t.sum(t.unsqueeze(seq_emb, 1) * all_emb, -1) # (batch, 100)
                seq_len = (seq > 0).cpu().numpy().sum(-1)
                h5, n5, h10, n10, h20, n20, h50, n50, gp_h20, gp_n20, gp_num= \
                    self.calc_res(all_scr.cpu().numpy(), all_ids.cpu().numpy(), pos.cpu().numpy(), seq_len, logPredict)
                ep_h5 += h5
                ep_n5 += n5
                ep_h10 += h10
                ep_n10 += n10
                ep_h20 += h20
                ep_n20 += n20
                ep_h50 += h50
                ep_n50 += n50
                for j in range(4):
                    group_h20[j] += gp_h20[j]
                    group_n20[j] += gp_n20[j]
                    group_num[j] += gp_num[j]
                log('Steps %d/%d: hr@5 = %.2f, ndcg@5 = %.2f, hr@10 = %.2f, ndcg@10 = %.2f, hr@20 = %.2f, ndcg@20 = %.2f, hr@50 = %.2f, ndcg@50 = %.2f ' % (i, steps, h5, n5, h10, n10, h20, n20, h50, n50), save=False, oneline=True)
                sys.stdout.flush()

        ep_h5 /= num
        ep_n5 /= num
        ep_h10 /= num 
        ep_n10 /= num 
        ep_h20 /= num 
        ep_n20 /= num 
        ep_h50 /= num 
        ep_n50 /= num 

        for i in range(4):
            group_h20[i] /= group_num[i]
            group_n20[i] /= group_num[i]

        ret = dict()
        ret['hr@10'] = ep_h10
        ret['ndcg@10'] = ep_n10
        ret['hr@5'] = ep_h5
        ret['ndcg@5'] = ep_n5
        ret['hr@20'] = ep_h20
        ret['ndcg@20'] = ep_n20
        ret['hr@50'] = ep_h50
        ret['ndcg@50'] = ep_n50

        print(f'Test result: h5={ep_h5:.4f} n5={ep_n5:.4f} h10={ep_h10:.4f} n10={ep_n10:.4f} h20={ep_h20:.4f} n20={ep_n20:.4f} h50={ep_h50:.4f} n50={ep_n50:.4f}')

        return ret

    def calc_res(self, scores, tst_ids, pos_ids, seq_len, logPredict):
        group_h20 = [0] * 4
        group_n20 = [0] * 4
        group_num = [0] * 4
        h5, n5, h10, n10, h20, n20, h50, n50 = [0] * 8
        for i in range(len(pos_ids)):
            ids_with_scores = list(zip(tst_ids[i], scores[i]))
            ids_with_scores = sorted(ids_with_scores, key=lambda x: x[1], reverse=True)
            if seq_len[i] < 5:
                group_num[0] += 1
            elif seq_len[i] >= 5 and seq_len[i] < 10:
                group_num[1] += 1
            elif seq_len[i] >= 10 and seq_len[i] < 20:
                group_num[2] += 1
            else:
                group_num[3] += 1
            shoot = list(map(lambda x: x[0], ids_with_scores[:5]))
            if pos_ids[i] in shoot:
                h5 += 1
                n5 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            shoot = list(map(lambda x: x[0], ids_with_scores[:10]))
            if pos_ids[i] in shoot:
                h10 += 1
                n10 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            shoot = list(map(lambda x: x[0], ids_with_scores[:20]))
            if pos_ids[i] in shoot:
                if seq_len[i] < 5:
                    group_h20[0] += 1
                    group_n20[0] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                elif seq_len[i] >= 5 and seq_len[i] < 10:
                    group_h20[1] += 1
                    group_n20[1] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                elif seq_len[i] >= 10 and seq_len[i] < 20:
                    group_h20[2] += 1
                    group_n20[2] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                else:
                    group_h20[3] += 1
                    group_n20[3] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                h20 += 1
                n20 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            shoot = list(map(lambda x: x[0], ids_with_scores[:50]))
            if pos_ids[i] in shoot:
                h50 += 1
                n50 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            if logPredict: 
                self.writePredictedItems(shoot)
        return h5, n5, h10, n10, h20, n20, h50, n50, group_h20, group_n20, group_num

    def save_history(self):
        if args.epoch == 0:
            return

        if not os.path.exists('./Models/'):
                os.makedirs('./Models/')

        if not os.path.exists('./History/'):
                os.makedirs('./History/')

        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'encoder': self.encoder,
            'recommender': self.recommender,
        }
        t.save(content, './Models/' + args.save_path + '.mod')

        log('Model Saved: %s' % args.save_path)

    def load_model(self):
        ckp = t.load('./Models/' + args.load_model + '.mod')
        self.encoder = ckp['encoder']
        self.recommender = ckp['recommender']
        self.opt = t.optim.Adam(
            [{"params": self.encoder.parameters()},
             {"params": self.recommender.parameters()}],
            lr=args.lr, weight_decay=0
        )

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)

        log('Model Loaded from ' + args.load_model)

    def writePredictedItems(self, data):
        with open('predicts.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)

if __name__ == '__main__':
    logger.saveDefault = True
    
    print(args)

    log('Start')
    handler = DataHandler()
    handler.load_data()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
