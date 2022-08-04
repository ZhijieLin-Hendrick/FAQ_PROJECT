from ..dataset_module.dataset import PadBatch, PairedSentenceDataset
from torch.utils.data import RandomSampler, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch, logging, tqdm, json
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def get_paired_dataIter(
    data_path:str, 
    tokz,
    device,
    maxSeqLen:int = 60,
    batchSize:int = 300,
    *args, **kwargs
):

    dataset = PairedSentenceDataset(data_path, maxSeqLen, tokz,*args, **kwargs)
    data_sampelr = RandomSampler(dataset)
    data_iter = DataLoader(dataset=dataset, batch_size=batchSize, sampler=data_sampelr, 
                           collate_fn=PadBatch(tokz.pad_token_id, device))

    return data_iter


def compute_kernel_and_bias(vecs: Union[List[np.ndarray], np.array]):

    if isinstance(vecs, list):
        vecs = np.concatenate(vecs, axis=0)
    
    mu = vecs.mean(axis=0, keepdims=0)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.dot(u, np.diag(1 / np.sqrt(s))))
    
    return -mu, W
    
def transform_and_normalize(vecs:Union[np.ndarray, List[np.ndarray]], kernel=None, bias=None):

    if isinstance(vecs, list):
        vecs = np.concatenate(vecs, axis=0)

    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)

    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)
    


def L2_norm(vecs):

    if isinstance(vecs, torch.Tensor):
        norms = (vecs**2).sum(dim=1, keepdims=True).sqrt()
        return vecs / torch.clamp(norms, 1e-8, )
    elif isinstance(vecs, np.ndarray):
        norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
        return vecs / np.clip(norms, 1e-8, np.inf)


def get_logger(filename, print2screen=True):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] \
>> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if print2screen:
        logger.addHandler(ch)
    return logger

def test(model, data_iter:DataLoader, save_sims_path:Optional[str]=None) -> Dict[str, float]:
    
    all_sims = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_iter):
            sents_1 = batch['sent1']
            sents_2 = batch['sent2']
            labels = batch['label']

            cls_pooler_sent1_vecs = model(**sents_1).pooler_output
            cls_pooler_sent2_vecs = model(**sents_2).pooler_output
            

            cls_pooler_sent1_vecs = L2_norm(cls_pooler_sent1_vecs)
            cls_pooler_sent2_vecs = L2_norm(cls_pooler_sent2_vecs)

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




def cal_metric(logits_outputs:torch.tensor, labels: torch.tensor):
    '''
    Input:
    :@param logits_outputs: (batchSize, n_labels)
    :@param labels: (batchSize)
    
    Output:
    result_metric: Dict[str, flaot]
    - Detailed: (acc, recall, precision, f1)
    '''

    if isinstance(labels, torch.Tensor):
        y_true = labels.detach().cpu().numpy()
    if isinstance(logits_outputs, torch.Tensor):
        y_pred = logits_outputs.argmax(dim=-1).detach().cpu().numpy()
    assert len(y_true) == len(y_pred)

    result_metrics = {}
    result_metrics['acc'] = accuracy_score(y_true, y_pred)
    result_metrics['f1'] = f1_score(y_true, y_pred)
    result_metrics['precision'] = precision_score(y_true, y_pred)
    result_metrics['recall'] = recall_score(y_true, y_pred)

    return result_metrics