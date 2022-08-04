import json
from itertools import accumulate
from re import L
from typing import List, Dict, Tuple, Optional, Union
import torch
from .__init__ import BaseTrainer
from ..loss_module.loss import SimCSELoss
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm 
from ..utils import util
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from typing import Optional
import numpy as np


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

class Trainer4SimCSE(BaseTrainer):

    def _get_loss(self, *args, **kwargs):
        return SimCSELoss()

    def _get_optimizer(self, lr:int=None, *args, **kwargs):
        if lr is not None:
            self.lr = lr
        return AdamW(self.model.parameters(), lr=self.lr)
    
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

    def _save_model(self):
        if self.save_path is None:
            self.save_path = './FAQ project/tmp_data/model_checkpoint/simbert_SimCSE'
        self.model.save_pretrained(self.save_path)

    def _load_model(self):
        pass

    def _align_sents(self, sents_1, sents_2):
        pad_token_id = self.tokz.pad_token_id

        batchSize = sents_1['input_ids'].shape[0]
        sent1_seqLen = sents_1['input_ids'].shape[1]
        sent2_seqLen = sents_2['input_ids'].shape[1]

        maxSeqLen = max(sent1_seqLen, sent2_seqLen)   
        if maxSeqLen > sent1_seqLen:
            # pad sents_1
            sents_1_input_pad = torch.full((batchSize, (maxSeqLen - sent1_seqLen)), pad_token_id).to(self.device)   # (batchSize, maxSeqLen-seqLen)
            sents_1_token_type_pad = torch.full((batchSize, (maxSeqLen - sent1_seqLen)), 0).to(self.device) # (batchSize, maxSeqLen-seqLen)
            sents_1_mask_pad = torch.full((batchSize, (maxSeqLen - sent1_seqLen)), 0).to(self.device)   # (batchSize, maxSeqLen-seqLen)

            sents_1['input_ids'] = torch.concat([sents_1['input_ids'], sents_1_input_pad], dim=-1)  # (batchSize, maxSeqLen)
            sents_1['token_type_ids'] = torch.concat([sents_1['token_type_ids'], sents_1_token_type_pad], dim=-1)   # (batchSize, maxSeqLen)
            sents_1['attention_mask'] = torch.concat([sents_1['attention_mask'], sents_1_mask_pad], dim=-1)     # (batchSize, maxSeqLen)

        if maxSeqLen > sent2_seqLen:
            # pad sents_2
            sents_2_input_pad = torch.full((batchSize, (maxSeqLen - sent2_seqLen)), pad_token_id).to(self.device)   # (batchSize, maxSeqLen-seqLen)
            sents_2_token_type_pad = torch.full((batchSize, (maxSeqLen - sent2_seqLen)), 0).to(self.device) # (batchSize, maxSeqLen-seqLen)
            sents_2_mask_pad = torch.full((batchSize, (maxSeqLen - sent2_seqLen)), 0).to(self.device)   # (batchSize, maxSeqLen-seqLen)

            sents_2['input_ids'] = torch.concat([sents_2['input_ids'], sents_2_input_pad], dim=-1)  # (batchSize, maxSeqLen)
            sents_2['token_type_ids'] = torch.concat([sents_2['token_type_ids'], sents_2_token_type_pad], dim=-1)   # (batchSize, maxSeqLen)
            sents_2['attention_mask'] = torch.concat([sents_2['attention_mask'], sents_2_mask_pad], dim=-1)     # (batchSize, maxSeqLen)

        return sents_1, sents_2

    def _merge_sents(self, sents_1, sents_2):

        sents_1, sents_2 = self._align_sents(sents_1, sents_2)  # ensure the seqLen be identical for both sentences by padding
        
        input_ids = torch.concat([sents_1['input_ids'], sents_2['input_ids']], dim=0)
        token_type_ids = torch.concat([sents_1['token_type_ids'], sents_2['token_type_ids']], dim=0)
        attention_mask = torch.concat([sents_1['attention_mask'], sents_2['attention_mask']], dim=0)

        return input_ids, token_type_ids, attention_mask

    def _train_epoch(self, epoch_i:int, *args, **kwargs):

        self.model.train()
        steps = 0
        accumulated_sample_size = 0
        accumulated_loss = 0 # accumulated total loss
        accumuated_batch_nums = 0 # number of accumulated batches
        
        TQDM = tqdm(enumerate(self.train_iter, desc=f"Train (epoch #{epoch_i})", total=len(self.train_iter)))
        for i, batch in TQDM:

            accumuated_batch_nums += 1
            
            sents_1 = batch['sent1'] #(batchSize, seqLen)
            sents_2 = batch['sent2'] #(batchSize, seqLen)
            labels = batch['label'] 

            input_ids, token_type_ids, attention_mask = self._merge_sents(sents_1, sents_2)
            sent_vecs_1 = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output #(batchSize*2, seqLen, hidden_nums)
            sent_vecs_2 = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output #(batchSize*2, seqLen, hidden_nums)

            batchSize, hidden_nums = sent_vecs_1.shape
            combined_sent_vecs = torch.concat([sent_vecs_1, sent_vecs_2], dim=-1) # (batchSize, hidden_nums*2)
            combined_sent_vecs = combined_sent_vecs.view(batchSize*2, hidden_nums) # (batchSize*2, hidden_nums)

            # gradient accumulation
            sample_size = combined_sent_vecs.shape[0] # sample_size = batchSize * 2
            avg_loss = self.criterion(combined_sent_vecs, reduction='mean')
            avg_loss_accumulated = avg_loss / self.BATCH_ACCUMULATED
            avg_loss_accumulated.backward()

            accumulated_loss += avg_loss.item() * sample_size
            accumulated_sample_size += sample_size

            TQDM.set_postfix({'train average loss': accumulated_loss / accumulated_sample_size,
                              'sample size': sample_size})

            # weight update
            if accumuated_batch_nums == self.BATCH_ACCUMULATED or i == len(self.train_iter)-1:

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                steps += 1
                # self.logger.info(f'train steps: {steps}, train average loss: {accumulated_loss / accumulated_sample_size}, \
                #                    accumulated_sample_size: {accumulated_sample_size}')

                accumuated_batch_nums = 0
                accumulated_loss = 0
                accumulated_sample_size = 0

            # evaluation
            if steps % self.eval_steps == 0:
                self._eval(steps)

    def _eval(self, steps:int, *args, **kwargs):
        '''
        Return the metrics
        - dev_loss
        '''
        self.model.eval()
        accumulated_loss = 0
        accumulated_sample_size = 0
        with torch.no_grad():
            for batch in self.dev_iter:
                sents_1 = batch['sent1'] #(batchSize, seqLen)
                sents_2 = batch['sent2'] #(batchSize, seqLen)
                labels = batch['label'] 

                input_ids, token_type_ids, attention_mask = self._merge_sents(sents_1, sents_2)
                sent_vecs_1 = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output #(batchSize, seqLen, hidden_nums)
                sent_vecs_2 = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output #(batchSize, seqLen, hidden_nums)

                batchSize, hidden_nums = sent_vecs_1.shape
                combined_sent_vecs = torch.concat([sent_vecs_1, sent_vecs_2], dim=-1) # (batchSize, hidden_nums*2)
                combined_sent_vecs = combined_sent_vecs.view(batchSize*2, hidden_nums) # (batchSize*2, hidden_nums)

                sample_size = combined_sent_vecs.shape[0] # sample_size = batchSize * 2
                sum_loss = self.criterion(combined_sent_vecs, reduction='sum')
                accumulated_loss += sum_loss
                accumulated_sample_size += sample_size
            
            dev_avg_loss = accumulated_loss / accumulated_sample_size
            self.logger.info(f'dev average loss: {dev_avg_loss}, sample size: {accumulated_sample_size}')
        
        # save model
        try:
            self.dev_best_loss
        except:
            self.dev_best_loss = float('inf')
        
        if dev_avg_loss < self.dev_best_loss:
            self.dev_best_loss = dev_avg_loss
            self._save_model()

        self._save_model()
        self.model.train()

    def fit(self):
        self.dev_best_loss = float('inf')
        for epoch in range(self.MAX_EPOCH):
            self._train_epoch(epoch)

class Trainer4SupSimCSE(Trainer4SimCSE):    # based on simbert

    def __init__(self, model=None, tokz=None, max_epoch: int = 1, batch_split: int = 1, dev_batch_split: int = 1, batch_accumulated: int = 1, warmup_steps: int = 1000, train_path: str = None, dev_path: str = None, batchSize: int = None, dev_batchSize: int = None, lr: float = 0.00001, dropout_rate: float = 0.3, maxSeqLen: int = 60, device=None, eval_steps: int = None, logger=None, save_path: str = None, early_stop: int = 10, *args, **kwargs):
        super().__init__(model, tokz, max_epoch, batch_split, dev_batch_split, batch_accumulated, warmup_steps, train_path, dev_path, batchSize, dev_batchSize, lr, dropout_rate, maxSeqLen, device, eval_steps, logger, save_path, early_stop, *args, **kwargs)
        self.train_iter = util.get_paired_dataIter(train_path, tokz, self.device, maxSeqLen, self.batchSize, onlyKeepPos_flag=True)
        self.dev_iter = util.get_paired_dataIter(dev_path, tokz, self.device, maxSeqLen, self.dev_batchSize)
        self.test_iter = util.get_paired_dataIter(dev_path, tokz, self.device, maxSeqLen, self.dev_batchSize)
        
    def _merge_sents(self, sents_1, sents_2):
        '''
        Ensure s_{2i} and s_{2i+1} are the match pair
        '''
        sents_1, sents_2 = self._align_sents(sents_1, sents_2)
        seqLen = sents_1['input_ids'].shape[1]

        input_ids = torch.concat([sents_1['input_ids'], sents_2['input_ids']], dim=-1) #(batchSize, seqLen * 2)
        input_ids = input_ids.view(-1, seqLen)
        token_type_ids = torch.concat([sents_1['token_type_ids'], sents_2['token_type_ids']], dim=-1) #(batchSize, seqLen * 2)
        token_type_ids = token_type_ids.view(-1, seqLen)
        attention_mask = torch.concat([sents_1['attention_mask'], sents_2['attention_mask']], dim=-1) #(batchSize, seqLen * 2)
        attention_mask = attention_mask.view(-1, seqLen)

        return input_ids, token_type_ids, attention_mask

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
            assert sum(labels).item() == len(labels)

            input_ids, token_type_ids, attention_mask = self._merge_sents(sents_1, sents_2) #(batchSize * 2, seqLen)
            sents_vecs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output #(batchSize * 2, seqLen, hidden_nums)
            sample_size = sents_vecs.shape[0] # sample_size = batchSize * 2
            avg_loss = self.criterion(sents_vecs, reduction='mean')
            avg_loss_accumulated = avg_loss / self.BATCH_ACCUMULATED
            avg_loss_accumulated.backward()

            TQDM.set_postfix({'train average loss': avg_loss.item(),
                              'sample size': sample_size})
            
            accumulated_loss = avg_loss.item() * sample_size
            accumulated_sample_size += sample_size

            # weight update
            if accumulated_batch_nums == self.BATCH_ACCUMULATED or i == len(self.train_iter)-1:

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                steps += 1
                accumulated_batch_nums = 0

                # evaluation
                if steps % self.eval_steps == 0:

                    logger.info(f"tot train step #{steps}, train avg loss: {accumulated_loss / accumulated_sample_size}, \
                    tot sample size: {accumulated_sample_size}")
                    accumulated_loss = 0
                    accumulated_sample_size = 0
                    
                    self._eval(steps)

            try:
                self.early_stop_count
            except:
                self.early_stop_count = 0

            if self.early_stop_count == self.early_stop:
                self.logger.info('Early stop')
                break

    def _eval(self, steps:int, *args, **kwargs):
        '''
        Return the metrics
        - dev_loss
        '''
        self.model.eval()
        accumulated_sample_size = len(self.dev_iter)
        with torch.no_grad():
            metric_result = test(self.model, self.dev_iter)
            self.logger.info(f">>> Step #{steps}, sample size: {accumulated_sample_size}, \
            dev auc: {metric_result['auc']}, dev spearman: {metric_result['spearman']}")
        
        # save model
        try:
            self.dev_best_spearman
        except:
            self.dev_best_spearman = -float('inf')
        
        try:
            self.early_stop_count
        except:
            self.early_stop_count = 0

        if metric_result['spearman'] > self.dev_best_spearman:
            self.dev_best_spearman = metric_result['spearman']
            self._save_model()
            self.early_stop_count = 0
        else: 
            self.early_stop_count += 1

        self.model.train()


    def fit(self):
        self.dev_best_loss = float('inf')
        for epoch in range(self.MAX_EPOCH):
            self._train_epoch(epoch)
            if self.early_stop_count == self.early_stop:
              self.logger.info('Early stop')
              break
        test_metrics = test(self.model, self.test_iter)
        self.logger.info(test_metrics)


if __name__ == '__main__':
    import os
    from transformers import BertTokenizer, BertModel, BertConfig

    model_name = "WangZeJun/simbert-base-chinese"
    logger_file_name = './FAQ project/log/sup_simcse_batchSize16.log'
    logger = util.get_logger(logger_file_name)


    logger.info(os.getcwd())
    logger.info(os.path.exists('./FAQ project/data/merged_qq_train.csv'))


    config = BertConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = 0.3
    model = BertModel.from_pretrained(model_name, config)
    model.cuda()
    logger.info('successfully load model')
    tokz = BertTokenizer.from_pretrained(model_name)
    
    trainer = Trainer4SupSimCSE(train_path='./FAQ project/data/merged_qq_train.csv',dev_path='./FAQ project/data/merged_qq_dev.csv', 
                             test_path='./FAQ project/data/merged_qq_test.csv',
                             lr=1e-5, tokz=tokz, batchSize=16, dev_batchSize=200, model=model, max_epoch=2, logger=logger,
                             eval_steps=300)
                        
    trainer.fit()