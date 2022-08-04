from torch.utils.data import Dataset
import pandas as pd
from typing import Union, List, Dict, Tuple
import torch

class PairedSentenceDataset(Dataset):

    def __init__(self, data_path:str, maxSeqLen:int, tokz, **kwargs):
        '''
        :params maxSeqLen: int
            - this one do not take cls and sep into consideration  
        '''
        super(PairedSentenceDataset, self).__init__()
        self.data_path = data_path
        self.maxSeqLen = maxSeqLen
        self.tokz = tokz
        if "onlyKeepPos_flag" in kwargs:
            self.onlyKeepPos_flag = kwargs['onlyKeepPos_flag']
        self.data_and_label = PairedSentenceDataset.make_dataset(data_path, maxSeqLen, tokz, **kwargs)
        
    @staticmethod
    def make_dataset(data_path:str, maxSeqLen:int, tokz, **kwargs):
        '''
        Detailed: 
            - this one would not return the padding data
        '''
        onlyKeepPos_flag = False
        if "onlyKeepPos_flag" in kwargs:
            onlyKeepPos_flag = kwargs["onlyKeepPos_flag"]
        data_and_label = [] # list of dictionary of pair sentences(sent1: tokz_sent1, sent2: tokz_sent2, label: label)
        with open(data_path, encoding='utf-8') as f:
            for line in f:

                if len(line.strip().split("\t")) < 3:
                    continue
                sent1, sent2, label = line.strip().split("\t")

                if int(float(label)) not in [0, 1]:
                    continue
                if onlyKeepPos_flag:
                    if int (float(label)) != 1:
                        continue
                
                # truncate
                sent1, sent2 = sent1[:maxSeqLen], sent2[:maxSeqLen]
                label = int(float(label))

                tokz_sent1 = dict(tokz(sent1))
                tokz_sent2 = dict(tokz(sent2))

                data_and_label.append({'sent1':tokz_sent1, 'sent2':tokz_sent2, 'label':label})
        return data_and_label

    
    def __len__(self):
        return len(self.data_and_label)
                

    def __getitem__(self, index):
        return self.data_and_label[index]
        

class PadBatch:
    
    def __init__(self, pad_id: int, device:str):
        self.pad_id = pad_id
        self.device = device

    def _pad_data(self, sents:List[Dict[str, List[str]]]):
        '''
        sents: represent sentence 1 or sentence 2
        '''
        max_len = max([len(s['input_ids']) for s in sents])
        res = dict()

        res['input_ids'] = torch.LongTensor([s['input_ids'] + [self.pad_id] * (max_len - len(s['input_ids'])) for s in sents])
        res['token_type_ids'] = torch.LongTensor([s['token_type_ids'] + [0] * (max_len - len(s['input_ids'])) for s in sents])
        res['attention_mask'] = torch.LongTensor([s['attention_mask'] + [0] * (max_len - len(s['input_ids'])) for s in sents])

        res['input_ids'] = res['input_ids'].to(self.device)
        res['token_type_ids'] = res['token_type_ids'].to(self.device)
        res['attention_mask'] = res['attention_mask'].to(self.device)

        return res

    def __call__(self, batch: List[Dict[str,  Union[Dict[str, List[str]],int]]]):
        '''
        batch: List[Dict[str, Union[Dict[str, List[int]]:(sent1/sent2), int:label]]]
        '''
        sents_1 = [d['sent1'] for d in batch]
        sents_2 = [d['sent2'] for d in batch]
        labels = [d['label'] for d in batch]

        pad_sents_1 = self._pad_data(sents_1)
        pad_sents_2 = self._pad_data(sents_2)
        pad_labels = torch.LongTensor(labels).to(self.device)
        

        return {'sent1': pad_sents_1, 'sent2': pad_sents_2, 'label': pad_labels}
        
if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader, RandomSampler
    tokz = BertTokenizer.from_pretrained('bert-base-chinese')
    
    device = torch.ones(1).cuda().device
    dataset = PairedSentenceDataset('./FAQ project/data/merged_qq_train.csv', 60, tokz, onlyKeepPos_flag=False)
    data_sampelr = RandomSampler(dataset)
    data_iter = DataLoader(dataset=dataset, batch_size=100, sampler=data_sampelr, 
                           collate_fn=PadBatch(tokz.pad_token_id, device))

    for batch in data_iter:
        print(batch['label'])
        break
