# evaluate model on the dataset 
## 1. directly use SimBERT and RoFormer-Sim to do a test
## 1.1 directly use cls pooling output [cause we use this one to train the model]
## 1.2 encoder

# 1. need model and dataset 
## 1.1 dataset: singleSentenceDataset
## 1.2 model: import model

from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from .utils import util
import torch
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from typing import Optional
from tqdm import tqdm



def test_with_whitening(model, data_iter:DataLoader, n_components:int=None, save_sims_path:Optional[str]=None):
    pair_vecs = {'sent1':[], 'sent2':[]}
    all_vecs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_iter):

            sents_1 = batch['sent1']
            sents_2 = batch['sent2']
            labels = batch['label']

            cls_pooler_sent1_vecs = model(**sents_1).pooler_output
            cls_pooler_sent2_vecs = model(**sents_2).pooler_output

            cls_pooler_sent1_vecs_np = cls_pooler_sent1_vecs.detach().cpu().numpy()
            cls_pooler_sent2_vecs_np = cls_pooler_sent2_vecs.detach().cpu().numpy()

            all_vecs.append(cls_pooler_sent1_vecs_np)
            all_vecs.append(cls_pooler_sent2_vecs_np)
            pair_vecs['sent1'].append(cls_pooler_sent1_vecs_np)
            pair_vecs['sent2'].append(cls_pooler_sent2_vecs_np)
            all_labels.extend(labels.detach().cpu().numpy().copy())
            
    bias, kernel = util.compute_kernel_and_bias(all_vecs)
    if n_components is not None:
        kernel = kernel[:, :n_components]

    pair_vecs['sent1'] = util.transform_and_normalize(pair_vecs['sent1'], kernel, bias)
    pair_vecs['sent2'] = util.transform_and_normalize(pair_vecs['sent2'], kernel, bias)
    if isinstance(all_labels, list):
        all_labels = np.array(all_labels)

    all_sims = (pair_vecs['sent1'] * pair_vecs['sent2']).sum(axis=1).flatten()
    corr_score = spearmanr(all_sims, all_labels)
    auc_score = roc_auc_score(all_labels, all_sims)

    print(f'The test dataset spearman score is: {corr_score}')
    print(f'The test dataset auc score is: {auc_score}')

    if save_sims_path is not None:
        res = {'label':all_labels.tolist(), 'sims': all_sims.tolist()}
        json.dump(res, open(save_sims_path, 'w'))

    return {'spearman': corr_score, 'auc': auc_score}


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

    return {'spearman': corr_score, 'auc': auc_score}
    

if __name__ == '__main__':
    from transformers import BertTokenizer, BertModel

    model_name = "WangZeJun/simbert-base-chinese"
    simbert = BertModel.from_pretrained('./FAQ project/tmp_data/model_checkpoint/simbert_SimCSE/')
    # simbert = BertModel.from_pretrained(model_name)
    tokz = BertTokenizer.from_pretrained(model_name)
    

    simbert.cuda()
    simbert.eval()
    device = simbert.device
    test_iter = util.get_paired_dataIter('./FAQ project/data/merged_qq_train.csv', tokz, device)

    test(simbert, test_iter ,save_sims_path='./FAQ project/tmp_data/simbert_test')

    # from roformer import RoFormerForCausalLM, RoFormerConfig
    # from transformers import BertTokenizer
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # pretrained_model = "junnyu/roformer_chinese_sim_char_base"
    # tokz = BertTokenizer.from_pretrained(pretrained_model)
    # config = RoFormerConfig.from_pretrained(pretrained_model)
    # config.is_decoder = True
    # config.eos_token_id = tokz.sep_token_id
    # config.pooler_activation = "linear"
    # roformer_sim = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
    # roformer_sim.to(device)
    # roformer_sim.eval()

    # test_iter = util.get_paired_dataIter('./data/merged_qq_test.csv', tokz, device)
    # test_with_whitening(roformer_sim, test_iter, 384, save_sims_path='./tmp_data/roformer_sim_whitening_test')
