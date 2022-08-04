from unittest import result

from numpy import isin
from .train4simcse import test, Trainer4SimCSE, Trainer4SupSimCSE
from ..utils import util
from .__init__ import BaseTrainer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..model_module import model
import torch
from transformers import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from ..model_module.model import SBERT
import json
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ..loss_module import loss

def test(model, data_iter:DataLoader, save_sims_path:Optional[str]=None):
    
    all_sims = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_iter):
            sents_1 = batch['sent1']
            sents_2 = batch['sent2']
            labels = batch['label']

            cls_pooler_sent1_vecs = model(**sents_1).pooler_output
            cls_pooler_sent2_vecs = model(**sents_2).pooler_output
            

            cls_pooler_sent1_vecs = util.L2_norm(cls_pooler_sent1_vecs)
            cls_pooler_sent2_vecs = util.L2_norm(cls_pooler_sent2_vecs)

            sims = (cls_pooler_sent1_vecs * cls_pooler_sent2_vecs).sum(dim=1)

            if sims.device.type != 'cpu':
                sims = sims.cpu()
                labels = labels.cpu()

            sims_np = sims.flatten().numpy()
            labels_np = labels.flatten().numpy()
            assert len(sims_np) == len(labels_np)

            all_sims.append(sims_np)
            all_labels.append(labels_np)

    all_sims = np.hstack(all_sims)
    all_labels = np.hstack(all_labels)
        
    corr_score = spearmanr(all_sims, all_labels)
    auc_score = roc_auc_score(all_labels, all_sims)


    print(f'The test dataset spearman score is: {corr_score}')
    print(f'The test dataset auc score is: {auc_score}')

    if save_sims_path is not None:
        res = {'label':all_labels.tolist(), 'sims': all_sims.tolist()}
        json.dump(res, open(save_sims_path, 'w'))

    return {'spearman': corr_score.correlation, 'auc': auc_score}

def cal_metric(logits_outputs:torch.tensor, labels: torch.tensor):
    '''
    Input:
    :@param logits_outputs: (batchSize, n_labels)
    :@param labels: (batchSize)
    
    Output:
    result_metric: Dict[str, flaot]
    - Detailed: (acc, recall, precision, f1)
    '''
    if isinstance(y_true, torch.Tensor):
        y_true = labels.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = logits_outputs.argmax(dim=-1).detach().cpu().numpy()
    assert len(y_true) == len(y_pred)

    result_metrics = {}
    result_metrics['acc'] = accuracy_score(y_true, y_pred)
    result_metrics['f1'] = f1_score(y_true, y_pred)
    result_metrics['prescision'] = precision_score(y_true, y_pred)
    result_metrics['recall'] = recall_score(y_true, y_pred)

    return result_metrics

class Trainer4SupSBERT(BaseTrainer):
    def __init__(self, model=None, tokz=None, max_epoch: int = 1, batch_split: int = 1, dev_batch_split: int = 1, batch_accumulated: int = 1, warmup_steps: int = 1000, train_path: str = None, dev_path: str = None, batchSize: int = None, dev_batchSize: int = None, lr: float = 0.00001, dropout_rate: float = 0.3, maxSeqLen: int = 60, device=None, eval_steps: int = None, logger=None, save_path: str = None, early_stop: int = 10, *args, **kwargs):
        super().__init__(model, tokz, max_epoch, batch_split, dev_batch_split, batch_accumulated, warmup_steps, train_path, dev_path, batchSize, dev_batchSize, lr, dropout_rate, maxSeqLen, device, eval_steps, logger, save_path, early_stop, *args, **kwargs)
        self.train_iter = util.get_paired_dataIter(train_path, tokz, self.device, maxSeqLen, self.batchSize)
        self.dev_iter = util.get_paired_dataIter(dev_path, tokz, self.device, maxSeqLen, self.dev_batchSize)
        self.test_iter = util.get_paired_dataIter(dev_path, tokz, self.device, maxSeqLen, self.dev_batchSize)

    def _get_lr_scheduler(self, warmup_steps:int=None, *args, **kwargs):
        
        if warmup_steps is not None:
            self.WARMUP_STEPS = warmup_steps
        TOT_TRAINING_STEPS = len(self.train_iter) * self.MAX_EPOCH // self.BATCH_ACCUMULATED
        WARMUP_STEPS = self.WARMUP_STEPS
        return get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = WARMUP_STEPS,
            num_training_steps = TOT_TRAINING_STEPS
        )

    def _get_loss(self):
        return nn.CrossEntropyLoss(reduction='mean')

    def _get_optimizer(self, lr:int=None, *args, **kwargs):
        if lr is not None:
            self.lr = lr
        return AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)   # need to add weight decay

    def _train_epoch(self, epoch_i: int, *args, **kwargs):
        self.model.train()
        
        steps = 0

        accumulated_sample_size = 0
        accumulated_loss = 0 # accumulated total loss
        accumulated_batch_nums = 0 # number of accumulated batches

        TQDM = tqdm(enumerate(self.train_iter), desc=f"Train (epoch #{epoch_i})", total=len(self.train_iter))
        for i, batch in TQDM:

            accumulated_batch_nums += 1

            sents_1 = batch['sent1'] #(batchSize, seqLen)
            sents_2 = batch['sent2'] #(batchSize, seqLen)
            labels = batch['label']  #(batchSize)
            sample_size = len(labels)

            logits_outputs = self.model(sents_1, sents_2)   # (batchSize, n_labels|2)
            avg_loss = self.criterion(logits_outputs, labels)

            # gradient accumulated
            avg_loss_accumulated = avg_loss / self.BATCH_ACCUMULATED
            avg_loss_accumulated.backward()

            # print train loss and avg 
            train_metrics:dict = util.cal_metric(logits_outputs, labels)


            TQDM.set_postfix({'sample_size': sample_size, 'train_avg_loss': avg_loss.item(), 'train_f1': train_metrics['f1']})

            accumulated_loss += avg_loss * sample_size
            accumulated_sample_size += sample_size

            if accumulated_batch_nums == self.BATCH_ACCUMULATED or i == len(self.train_iter)-1:

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                steps += 1
                accumulated_batch_nums = 0

                # evaluation
                if steps % self.eval_steps == 0:

                    self.logger.info(f'tot train step #{steps}, train_avg_loss: {accumulated_loss / accumulated_sample_size}, tot_sample_size: {accumulated_sample_size}')
                    accumulated_loss = 0
                    accumulated_sample_size = 0

                    self._eval(epoch_i, steps)

            try:
                self.early_stop_count
            except:
                self.early_stop_count = 0

            if self.early_stop_count == self.early_stop:
                self.logger.info('Early stop')
                break

    def _eval(self, epoch_i:int=None, steps: int=None, use_dev:bool=True, *args, **kwargs):
        
        self.model.eval()
        
        accumulated_sample_size = 0
        accumulated_loss = 0

        all_logits_outputs = []
        all_labels = []

        if use_dev:
            eval_iter = self.dev_iter
        else:
            eval_iter = self.test_iter

        with torch.no_grad():
            for batch in eval_iter:

                sents_1 = batch['sent1'] #(batchSize, seqLen)
                sents_2 = batch['sent2'] #(batchSize, seqLen)
                labels = batch['label']  #(batchSize)
                sample_size = len(labels) 

                logits_outputs = self.model(sents_1, sents_2)   # (batchSize, n_labels|2)
                avg_loss = self.criterion(logits_outputs, labels)

                accumulated_sample_size += sample_size
                accumulated_loss += (avg_loss.item() * sample_size)

                all_logits_outputs.append(logits_outputs)
                all_labels.append(labels)
        
        all_logits_outputs = torch.concat(all_logits_outputs, dim=0)
        all_labels = torch.concat(all_labels, dim=0)

        assert len(all_logits_outputs.shape) == 2 and all_logits_outputs.shape[0] == accumulated_sample_size
        eval_metrics = util.cal_metric(all_logits_outputs, all_labels)
        eval_avg_loss = accumulated_loss / accumulated_sample_size

        if use_dev:
            self.logger.info(f"Dev: epoch #{epoch_i}, steps #{steps}, dev_avg_loss: {eval_avg_loss}, dev_acc: {eval_metrics['acc']}, dev_f1: {eval_metrics['f1']}, dev_precision: {eval_metrics['precision']}, dev_recall: {eval_metrics['recall']}, early_stop: {self.early_stop_count}/{self.early_stop}")

            # save model
            try:
                self.dev_best_f1
            except:
                self.dev_best_f1 = -float('inf')
            
            try:
                self.early_stop_count
            except:
                self.early_stop_count = 0

            if eval_metrics['f1'] > self.dev_best_f1:
                self.dev_best_f1 = eval_metrics['f1']
                self._save_model()
                self.early_stop_count = 0
            else: 
                self.early_stop_count += 1
        
        else:
            self.logger.info(f"Test: test_avg_loss: {eval_metrics}, test_acc: {eval_metrics['acc']}, test_f1: {eval_metrics['f1']}, test_precision: {eval_metrics['precision']}, test_recall: {eval_metrics['recall']}")

    def _save_model(self):
        self.model.save(self.save_path)

    def fit(self):

        for epoch in range(self.MAX_EPOCH):
            self._train_epoch(epoch)
            if self.early_stop_count == self.early_stop:
              self.logger.info('Early stop')
              break
        
class Trainer4SupBERT_RDropout(Trainer4SupSBERT):

    def _get_loss(self):
        '''
        Only support mean metrics
        '''
        return loss.extended_with_Rdropout(nn.CrossEntropyLoss, alpha=4)()

    def _merge_outputs(self, logits_outputs_1:torch.Tensor, logits_outputs_2:torch.Tensor):
        '''
        Ensure s_{2*i} and s_{2i+1} are different logits_outputs from the same sentence pair
        '''
        
        cur_batchSize, n_labels = logits_outputs_1.shape
        combined_logits_outputs = torch.concat([logits_outputs_1, logits_outputs_2], dim=-1)    # (batchSize, n_labels*2 | 2*2)
        combined_logits_outputs = combined_logits_outputs.view(cur_batchSize*2, n_labels)   # (batchSize*2, n_labels)

        return combined_logits_outputs
    
    def _train_epoch(self, epoch_i:int, *args, **kwargs):
        self.model.train()
        
        steps = 0

        accumulated_sample_size = 0
        accumulated_loss = 0 # accumulated total loss
        accumulated_batch_nums = 0 # number of accumulated batches

        TQDM = tqdm(enumerate(self.train_iter), desc=f"Train (epoch #{epoch_i})", total=len(self.train_iter))
        for i, batch in TQDM:

            accumulated_batch_nums += 1

            sents_1 = batch['sent1'] #(batchSize, seqLen)
            sents_2 = batch['sent2'] #(batchSize, seqLen)
            labels = batch['label']  
            sample_size = len(labels)

            logits_outputs_1 = self.model(sents_1, sents_2)   # (batchSize, n_labels|2)
            logits_outputs_2 = self.model(sents_1, sents_2)   # (batchSize, n_labels|2)
            combined_logits_outputs = self._merge_outputs(logits_outputs_1, logits_outputs_2)   # (batchSize*2, n_labels|2)
            combined_labels = labels.repeat_interleave(2, dim=0)    # (batchSize*2, )

            avg_loss = self.criterion(combined_logits_outputs, combined_labels)  # only return average loss

            # gradient accumulated
            avg_loss_accumulated = avg_loss / self.BATCH_ACCUMULATED
            avg_loss_accumulated.backward()

            # print train loss and avg
            train_metrics:dict = util.cal_metric(combined_logits_outputs, combined_labels)

            TQDM.set_postfix({'sample_size': sample_size, 'train_avg_loss': avg_loss.item(), 'train_f1': train_metrics['f1']})

            accumulated_loss += avg_loss * sample_size
            accumulated_sample_size += sample_size

            if accumulated_batch_nums == self.BATCH_ACCUMULATED or i == len(self.train_iter)-1:

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                steps += 1
                accumulated_batch_nums = 0

                # evaluation
                if steps % self.eval_steps == 0:
                    self.logger.info(f'tot train step #{steps}, train_avg_loss: {accumulated_loss / accumulated_sample_size}, tot_sample_size: {accumulated_sample_size}')
                    accumulated_loss = 0
                    accumulated_sample_size = 0

                    self._eval(epoch_i, steps)
            
            try:
                self.early_stop_count
            except:
                self.early_stop_count = 0

            if self.early_stop_count == self.early_stop:
                self.logger.info('Early stop')
                break
    
    def _eval(self, epoch_i: int = None, steps: int = None, use_dev: bool = True, *args, **kwargs):

        self.model.eval()

        accumulated_sample_size = 0
        accumulated_loss = 0

        all_logits_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dev_iter:

                sents_1 = batch['sent1'] #(batchSize, seqLen)
                sents_2 = batch['sent2'] #(batchSize, seqLen)
                labels = batch['label']  #(batchSize)
                sample_size = len(labels) 

                logits_outputs_1 = self.model(sents_1, sents_2)   # (batchSize, n_labels|2)
                logits_outputs_2 = self.model(sents_1, sents_2)   # (batchSize, n_labels|2)
                combined_logits_outputs = self._merge_outputs(logits_outputs_1, logits_outputs_2)   # (batchSize*2, n_labels|2)
                combined_labels = labels.repeat_interleave(2, dim=0)    # (batchSize*2, )

                avg_loss = self.criterion(combined_logits_outputs, combined_labels)  # only return average loss
                
                accumulated_sample_size += sample_size
                accumulated_loss += (avg_loss.item() * sample_size)

                all_logits_outputs.append(combined_logits_outputs)
                all_labels.append(combined_labels)

        all_logits_outputs = torch.concat(all_logits_outputs, dim=0)
        all_labels = torch.concat(all_labels, dim=0)
        
        assert len(all_logits_outputs.shape) == 2 and all_logits_outputs.shape[0] == accumulated_sample_size*2
        dev_metrics = util.cal_metric(all_logits_outputs, all_labels)
        dev_avg_loss = accumulated_loss / accumulated_sample_size

        self.logger.info(f"Dev: epoch #{epoch_i}, steps #{steps}, dev_avg_loss: {dev_avg_loss}, dev_acc: {dev_metrics['acc']}, dev_f1: {dev_metrics['f1']}, dev_precision: {dev_metrics['precision']}, dev_recall: {dev_metrics['recall']}, early_stop: {self.early_stop_count}/{self.early_stop}")

        # save model
        try:
            self.dev_best_f1
        except:
            self.dev_best_f1 = -float('inf')
        
        try:
            self.early_stop_count
        except:
            self.early_stop_count = 0

        if dev_metrics['f1'] > self.dev_best_f1:
            self.dev_best_f1 = dev_metrics['f1']
            self._save_model()
            self.early_stop_count = 0
        else: 
            self.early_stop_count += 1
    
    def fit(self):

        for epoch in range(self.MAX_EPOCH):
            self._train_epoch(epoch)
            if self.early_stop_count == self.early_stop:
              self.logger.info('Early stop')
              break
    

class Trainer4SupBERT_RDropout_Sent(Trainer4SupBERT_RDropout):

    def _get_loss(self):
        '''
        Only support mean metrics
        '''
        return loss.extended_with_Rdropout_sent(nn.CrossEntropyLoss, alpha=4)()
    
    def _merge_sents_vecs(self, sents_vecs1:torch.Tensor, sents_vecs2:torch.Tensor):
        
        batchSize, hidden_nums = sents_vecs1.shape
        combined_sents_vecs = torch.concat([sents_vecs1, sents_vecs2], dim=-1)
        combined_sents_vecs = combined_sents_vecs.view(-1, hidden_nums) # (batchSize*2, hidden_nums)
        return combined_sents_vecs

    def _train_epoch(self, epoch_i:int, *args, **kwargs):
        self.model.train()
        
        steps = 0

        accumulated_sample_size = 0
        accumulated_loss = 0 # accumulated total loss
        accumulated_batch_nums = 0 # number of accumulated batches

        TQDM = tqdm(enumerate(self.train_iter), desc=f"Train (epoch #{epoch_i})", total=len(self.train_iter))
        for i, batch in TQDM:

            accumulated_batch_nums += 1

            sents_1 = batch['sent1'] #(batchSize, seqLen)
            sents_2 = batch['sent2'] #(batchSize, seqLen)
            labels = batch['label']  
            sample_size = len(labels)

            sents1_vecs1, sents2_vecs1, logits_outputs_1 = self.model(sents_1, sents_2, with_sents_vecs=True)   # (batchSize, hidden_nums), (batchSize, hidden_nums), (batchSize, n_labels|2) 
            sents1_vecs2, sents2_vecs2, logits_outputs_2 = self.model(sents_1, sents_2, with_sents_vecs=True)   # (batchSize, hidden_nums), (batchSize, hidden_nums), (batchSize, n_labels|2)

            combined_sents_vecs1 = self._merge_sents_vecs(sents1_vecs1, sents2_vecs1)  # (batchSize * 2, hidden_nums) 第一次过dropout
            combined_sents_vecs2 = self._merge_sents_vecs(sents1_vecs2, sents2_vecs2)  # (batchSize * 2, hidden_nums) 第二次过dropout
            combined_logits_outputs = self._merge_outputs(logits_outputs_1, logits_outputs_2)   # (batchSize*2, n_labels|2)
            combined_labels = labels.repeat_interleave(2, dim=0)    # (batchSize*2, )

            # criterion中对于需要对于mean_kld*2当作单个部分，直接的mean_kld是sum_kld / (2*batchSize)，这个是不合理的
            avg_loss = self.criterion(combined_sents_vecs1, combined_sents_vecs2, combined_logits_outputs, combined_labels)
            
            # gradient accumulated
            avg_loss_accumulated = avg_loss / self.BATCH_ACCUMULATED
            avg_loss_accumulated.backward()

            # print train loss and avg
            train_metrics:dict = util.cal_metric(combined_logits_outputs, combined_labels)

            TQDM.set_postfix({'sample_size': sample_size, 'train_avg_loss': avg_loss.item(), 'train_f1': train_metrics['f1']})

            accumulated_loss += avg_loss * sample_size
            accumulated_sample_size += sample_size

            if accumulated_batch_nums == self.BATCH_ACCUMULATED or i == len(self.train_iter)-1:

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                steps += 1
                accumulated_batch_nums = 0

                # evaluation
                if steps % self.eval_steps == 0:
                    self.logger.info(f'tot train step #{steps}, train_avg_loss: {accumulated_loss / accumulated_sample_size}, tot_sample_size: {accumulated_sample_size}')
                    accumulated_loss = 0
                    accumulated_sample_size = 0

                    self._eval(epoch_i, steps)
            
            try:
                self.early_stop_count
            except:
                self.early_stop_count = 0

            if self.early_stop_count == self.early_stop:
                self.logger.info('Early stop')
                break
    
    def _eval(self, epoch_i: int = None, steps: int = None, use_dev: bool = True, *args, **kwargs):

        self.model.eval()

        accumulated_sample_size = 0
        accumulated_loss = 0

        all_logits_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dev_iter:

                sents_1 = batch['sent1'] #(batchSize, seqLen)
                sents_2 = batch['sent2'] #(batchSize, seqLen)
                labels = batch['label']  #(batchSize)
                sample_size = len(labels) 


                sents1_vecs1, sents2_vecs1, logits_outputs_1 = self.model(sents_1, sents_2, with_sents_vecs=True)   # (batchSize, hidden_nums), (batchSize, hidden_nums), (batchSize, n_labels|2) 
                sents1_vecs2, sents2_vecs2, logits_outputs_2 = self.model(sents_1, sents_2, with_sents_vecs=True)   # (batchSize, hidden_nums), (batchSize, hidden_nums), (batchSize, n_labels|2)

                combined_sents_vecs1 = self._merge_sents_vecs(sents1_vecs1, sents2_vecs1)  # (batchSize * 2, hidden_nums) 第一次过dropout
                combined_sents_vecs2 = self._merge_sents_vecs(sents1_vecs2, sents2_vecs2)  # (batchSize * 2, hidden_nums) 第二次过dropout
                combined_logits_outputs = self._merge_outputs(logits_outputs_1, logits_outputs_2)   # (batchSize*2, n_labels|2)
                combined_labels = labels.repeat_interleave(2, dim=0)    # (batchSize*2, )

                avg_loss = self.criterion(combined_sents_vecs1, combined_sents_vecs2, combined_logits_outputs, combined_labels)  # only return average loss
                
                accumulated_sample_size += sample_size
                accumulated_loss += (avg_loss.item() * sample_size)

                all_logits_outputs.append(combined_logits_outputs)
                all_labels.append(combined_labels)

        all_logits_outputs = torch.concat(all_logits_outputs, dim=0)
        all_labels = torch.concat(all_labels, dim=0)
        
        assert len(all_logits_outputs.shape) == 2 and all_logits_outputs.shape[0] == accumulated_sample_size*2
        dev_metrics = util.cal_metric(all_logits_outputs, all_labels)
        dev_avg_loss = accumulated_loss / accumulated_sample_size

        self.logger.info(f"Dev: epoch #{epoch_i}, steps #{steps}, dev_avg_loss: {dev_avg_loss}, dev_acc: {dev_metrics['acc']}, dev_f1: {dev_metrics['f1']}, dev_precision: {dev_metrics['precision']}, dev_recall: {dev_metrics['recall']}, early_stop: {self.early_stop_count}/{self.early_stop}")

        # save model
        try:
            self.dev_best_f1
        except:
            self.dev_best_f1 = -float('inf')
        
        try:
            self.early_stop_count
        except:
            self.early_stop_count = 0

        if dev_metrics['f1'] > self.dev_best_f1:
            self.dev_best_f1 = dev_metrics['f1']
            self._save_model()
            self.early_stop_count = 0
        else: 
            self.early_stop_count += 1
    

class Trainer4SupBERT_UnSupSimCSE(Trainer4SupBERT_RDropout_Sent):

    def _get_loss(self):
        '''
        Only support mean metrics
        '''
        return loss.extended_with_unsup_simcse(nn.CrossEntropyLoss, alpha=4)()



if __name__ == '__main__':
    from transformers import BertConfig, BertTokenizer, BertModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = "WangZeJun/simbert-base-chinese"
    logger_file_name = './FAQ project/log/sup_sbert_rdropoutUnSupSimCSE_simbert_batchSize16.log'
    logger = util.get_logger(logger_file_name)

    config = BertConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = 0.3
    config.n_labels = 2
    simbert_model = SBERT(config, model_name)
    simbert_model.to(device)
    tokz = BertTokenizer.from_pretrained(model_name)
    logger.info('successfully load model and tokenizer')

    # trainer = Trainer4SupBERT_RDropout_Sent(train_path='./FAQ project/data/merged_qq_train.csv',dev_path='./FAQ project/data/merged_qq_dev.csv', 
    #                          test_path='./FAQ project/data/merged_qq_test.csv', device=device,
    #                          lr=1e-5, tokz=tokz, batchSize=16, dev_batchSize=200, model=simbert_model, max_epoch=2, logger=logger,
    #                          eval_steps=200, save_path='./FAQ project/tmp_data/model_checkpoint/supSBERT_RDropoutSent_simbert_batchSize16.ckpt',
    #                          early_stop=15)

    trainer = Trainer4SupBERT_UnSupSimCSE(train_path='./FAQ project/data/merged_qq_train.csv',dev_path='./FAQ project/data/merged_qq_dev.csv', 
                             test_path='./FAQ project/data/merged_qq_test.csv', device=device,
                             lr=1e-5, tokz=tokz, batchSize=16, dev_batchSize=200, model=simbert_model, max_epoch=2, logger=logger,
                             eval_steps=200, save_path='./FAQ project/tmp_data/model_checkpoint/supSBERT_RDropoutUnSupSimCSE_simbert_batchSize16.ckpt',
                             early_stop=15)

    trainer.fit()
    
    tokz = BertTokenizer.from_pretrained(model_name)
    simbert_model = SBERT.load('./FAQ project/tmp_data/model_checkpoint/supSBERT_RDropoutUnSupSimCSE_simbert_batchSize16.ckpt', device)
    simbert_model.eval()
    test_iter = util.get_paired_dataIter('./FAQ project/data/merged_qq_test.csv', tokz, device, 60, 16)
    
    test_metrics = test(simbert_model.encoder, test_iter, f'./FAQ project/tmp_data/sup_sbert_rdropoutUnSupSimCSE_simbert_batchSize16_test')
    logger.info(f"Test, test_auc: {test_metrics['auc']}, test_spearman: {test_metrics['spearman']}")
