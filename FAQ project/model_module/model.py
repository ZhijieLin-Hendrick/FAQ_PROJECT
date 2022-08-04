from torch import nn
from transformers import BertConfig, BertPreTrainedModel, BertModel, RobertaForCausalLM, BertTokenizer
from roformer import RoFormerConfig, RoFormerPreTrainedModel, RoFormerForCausalLM, RoFormerModel
import torch


# 还是要用from_pretrained来载入model
# save_model：则是直接利用上torch.save(model.state_dict(), filename)就可以了



class SBERT(nn.Module):

    def __init__(
        self,
        config, 
        pretrained_model_path:str=None,
    ):
        super().__init__()
        self.config = config
        self.encoder = BertModel(config)
        if pretrained_model_path is not None:
            self.encoder = BertModel.from_pretrained(pretrained_model_path, config=self.config)
        self.n_labels = config.n_labels # default as 2 classes (pos or neg)
        self.classifier = nn.Linear(config.hidden_size*3, self.n_labels, bias=False)

    def forward(self, sents_1, sents_2, with_sents_vecs:bool = False):
        '''
        Input:
        :param sents_1/2: Dict[str, tensor]
        Ouput:
        output_logit: (batchSize, n_labels|2)
        '''
        sents_vecs_1 = self.encoder(**sents_1).pooler_output   # (batchSize, hidden_nums)
        sents_vecs_2 = self.encoder(**sents_2).pooler_output   # (batchSize, hidden_nums)

        # merge
        sents_vecs_1_minus_2 = (sents_vecs_1 - sents_vecs_2).abs()
        sents_vecs_1_and_2 = torch.concat([sents_vecs_1, sents_vecs_1, sents_vecs_1_minus_2], dim=-1) # (batchSize, 3*hidden_nums)

        # clf
        output_logits = self.classifier(sents_vecs_1_and_2) # (batchSize, n_labels|2)

        if not with_sents_vecs:
            return output_logits
        else:
            return sents_vecs_1, sents_vecs_2, output_logits
        

    def save(self, filepath:str):
        params = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath:str, device=None):
        params = torch.load(filepath)
        model = SBERT(params['config'])
        model.load_state_dict(params['state_dict'])
        if device is not None:
            model = model.to(device)
        return model


class SRoformer(nn.Module):

    def __init__(
        self,
        config,
        pretrained_model_path:str=None,
    ):
        super().__init__()
        self.config = config
        self.encoder = RoFormerForCausalLM(config)
        if pretrained_model_path is not None:
            self.encoder = RoFormerForCausalLM.from_pretrained(pretrained_model_path, config=self.config)
        self.n_labels = config.n_labels
        self.classifier = nn.Linear(config.hidden_size*3, self.n_labels, bias=False)

    def forward(self, sents_1, sents_2, with_sents_vecs:bool = False):
        '''
        Input:
        :param sents_1/2: Dict[str, tensor]
        Ouput:
        output_logit: (batchSize, n_labels|2)
        '''
        sents_vecs_1 = self.encoder(**sents_1).pooler_output   # (batchSize, hidden_nums)
        sents_vecs_2 = self.encoder(**sents_2).pooler_output   # (batchSize, hidden_nums)

        # merge
        sents_vecs_1_minus_2 = (sents_vecs_1 - sents_vecs_2).abs()
        sents_vecs_1_and_2 = torch.concat([sents_vecs_1, sents_vecs_1, sents_vecs_1_minus_2], dim=-1) # (batchSize, 3*hidden_nums)

        # clf
        output_logits = self.classifier(sents_vecs_1_and_2) # (batchSize, n_labels|2)

        if not with_sents_vecs:
            return output_logits
        else:
            return sents_vecs_1, sents_vecs_2, output_logits
        

    def save(self, filepath:str):
        params = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath:str, device=None):
        params = torch.load(filepath)
        model = SRoformer(params['config'])
        model.load_state_dict(params['state_dict'])
        if device is not None:
            model = model.to(device)
        return model



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.ones(1).cuda().device
    # pretrained_model = "junnyu/roformer_chinese_sim_char_base"
    # tokz = BertTokenizer.from_pretrained(pretrained_model)
    # config = RoFormerConfig.from_pretrained(pretrained_model)
    # config.is_decoder = True
    # config.eos_token_id = tokz.sep_token_id
    # config.pooler_activation = "linear"    
    # config.n_labels = 3
    # Sent_roformerSim = SRoformer(config, pretrained_model)

    # Sent_roformerSim.save('./model_checkpoint/model2.ckpt')
    # print('save successfully')

    # Sent_roformerSim = SRoformer.load('./model_checkpoint/model2.ckpt', device)
    # print('load successfully')


    # from ..utils import util
    # train_iter = util.get_paired_dataIter('./FAQ project/data/merged_qq_test.csv', tokz, device, 60, 5)
    # for batch in train_iter:
    #     break
    # output = Sent_roformerSim(batch['sent1'],batch['sent2'])
    # print(output.shape)

    #########################################

    model_name = "WangZeJun/simbert-base-chinese"
    # model_name = 'bert-base-chinese'
    tokz = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = 0.3
    config.n_labels = 3
    model = SBERT(config, model_name)
    model = model.to(device)


    model.save('./model_checkpoint/model1.ckpt')
    print('save successfully')

    model = SBERT.load('./model_checkpoint/model1.ckpt', device)
    print('load successfully')


    from ..utils import util
    train_iter = util.get_paired_dataIter('./FAQ project/data/merged_qq_test.csv', tokz, device, 60, 5)
    for batch in train_iter:
        break
    output = model(batch['sent1'], batch['sent2'], with_sents_vecs=True)
    # print(output.shape)
    print([i.shape for i in output])

    #################
    # 用自定义（load和save；如果没有提供本地的load_path，那么就直接用云端的；否则用本地的）
    

    