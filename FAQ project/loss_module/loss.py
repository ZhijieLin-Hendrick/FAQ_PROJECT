from re import sub
import torch
from ..utils import util
from torch import kl_div, nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn import functional as F
from typing import Optional, Callable, List



class SimCSELoss(CrossEntropyLoss):

    def _create_sim_labels(self, sents_vecs):

        sample_nums = sents_vecs.shape[0]   # sample_nums = batchSize * 2
        idxs = torch.arange(sample_nums)
        labels = torch.hstack([i for i1_i2 in zip(idxs[1::2], idxs[::2]) for i in i1_i2])
        labels = labels.to(sents_vecs.device)
        
        return labels

    def _cal_masked_sent_sims(self, sents_vecs):

        sents_vecs = util.L2_norm(sents_vecs) # (batchSize*2, hidden_nums)
        sents_sims = torch.matmul(sents_vecs, sents_vecs.T) 
        sents_sims = sents_sims * 20

        sample_nums = sents_sims.shape[0]
        eye_mask = torch.eye(sample_nums) * -1e12
        eye_mask = eye_mask.to(sents_sims.device)
        masked_sents_sims = sents_sims + eye_mask

        return masked_sents_sims

    def forward(
        self, 
        sents_vecs:torch.Tensor,
        reduction:int = 'mean',
    ):
        '''
        sents_vecs: (batchSize*2, hidden_nums)

        - Detailed:
            - s_{2j} and s_{2j+1} are the similar ones
        '''

        masked_sents_sims = self._cal_masked_sent_sims(sents_vecs) # (batchSize*2, hidden_nums)
        labels = self._create_sim_labels(sents_vecs) # (batchSize*2, ) [constractive learning]
        self.reduction = reduction
        loss = super(SimCSELoss, self).forward(masked_sents_sims, labels)
        return loss
        

## define a new SimCSELoss for batch split
### 1. for sents_vecs， we need to split the sentes_vecs into

class SimCSELoss2(CrossEntropyLoss):

    def _create_sim_labels(self, sents_vecs):

        sample_nums = sents_vecs.shape[0]   # sample_nums = batchSize * 2
        idxs = torch.arange(sample_nums)
        labels = torch.hstack([i for i1_i2 in zip(idxs[1::2], idxs[::2]) for i in i1_i2])
        labels = labels.to(sents_vecs.device)
        return labels

    def _cal_masked_sent_sims(self, sub_sents_vecs, sents_vecs):
        '''
        :@param sub_sents_vecs: (subSampleSize, hidden_nums)
        :@param sents_vecs: (batchSize*2 | sampleSize, hidden_nums)

        Return:
        sub_masked_sents_sims: (subSampleSize, hidden_nums)
        '''
        
        sub_sents_sims = torch.matmul(sub_sents_vecs, sents_vecs.T) 
        sub_sample_size, batch_size = sub_sents_sims.shape
        
        eye_mask = torch.eye(sub_sample_size) * -1e12   #(subSampleSize, subSampleSize)
        mask = torch.concat([eye_mask, torch.zeros((sub_sample_size, (batch_size - sub_sample_size)))], dim=1)  #(subSampleSize, batchSize)
        mask = mask.to(sub_sents_sims.device)

        sub_masked_sents_sims = sub_sents_sims + mask
        return sub_masked_sents_sims

    def forward(
        self, 
        sents_vecs:torch.Tensor,
        batch_split:int = 2,
        reduction:str = 'none',
    ):
        '''
        sents_vecs: (batchSize*2, hidden_nums)

        - Detailed:
            - s_{2j=1} and s_{2j} are the similar ones
        '''
        self.reduction = 'none'
        sample_size = sents_vecs.shape[0]   # sample_size == batchSize * 2
        assert (sample_size % batch_split == 0) and (sample_size // batch_split % 2 ==0)    
        # sents_vecs could be deviced into each part with the same number in it
        # In each part, the number of sample sould even rather odd
        
        tot_labels = self._create_sim_labels(sents_vecs)    # (batchSize, )
        sents_vecs = util.L2_norm(sents_vecs)   # (batchSize*2, hidden_nums)

        loss = []
        for i in range(0, sample_size, batch_split):
            start = int(i)
            end = int(i+batch_split)
            sub_sents_vecs = sents_vecs[start:end]
            sub_masked_sent_sims = self._cal_masked_sent_sims(sub_sents_vecs, sents_vecs)   #(subSampleSize, batchSize)
            sub_labels = tot_labels[start:end]    #(subSampleSize, )
            sub_loss = super(SimCSELoss2, self).forward(sub_masked_sent_sims, sub_labels)   # (subSampleSize, )
            loss.append(sub_loss)

        assert end == sample_size

        loss = torch.concat(loss, dim=0)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()

class SimCSELoss1(CrossEntropyLoss):

    def _create_sim_labels(self, sents_vecs):

        sample_nums = sents_vecs.shape[0]   # sample_nums = batchSize * 2
        idxs = torch.arange(sample_nums)
        labels = torch.hstack([i for i1_i2 in zip(idxs[1::2], idxs[::2]) for i in i1_i2])
        labels = labels.to(sents_vecs.device)
        return labels

    def _cal_masked_sent_sims(self, sub_sents_vecs, sents_vecs):
        '''
        :@param sub_sents_vecs: (subSampleSize, hidden_nums)
        :@param sents_vecs: (batchSize*2 | sampleSize, hidden_nums)

        Return:
        sub_masked_sents_sims: (subSampleSize, hidden_nums)
        '''
        
        sub_sents_sims = torch.matmul(sub_sents_vecs, sents_vecs.T) 
        sub_sample_size, batch_size = sub_sents_sims.shape
        
        eye_mask = torch.eye(sub_sample_size) * -1e12   #(subSampleSize, subSampleSize)
        mask = torch.concat([eye_mask, torch.zeros((sub_sample_size, (batch_size - sub_sample_size)))], dim=1)  #(subSampleSize, batchSize)
        mask = mask.to(sub_sents_sims.device)

        sub_masked_sents_sims = sub_sents_sims + mask
        return sub_masked_sents_sims

    def forward(
        self, 
        start:int,
        end:int,
        sents_vecs:torch.Tensor,
        batch_split:int = 2,
        reduction:str = 'none',
    ):
        '''
        sents_vecs: (batchSize*2, hidden_nums)

        - Detailed:
            - s_{2j=1} and s_{2j} are the similar ones
        '''
        self.reduction = 'none'
        tot_labels = self._create_sim_labels(sents_vecs)    # (batchSize, )
        sents_vecs = util.L2_norm(sents_vecs)   # (batchSize*2, hidden_nums)

        sub_sents_vecs = sents_vecs[start:end, :]
        sub_labels = tot_labels[start:end, :]

        sub_masked_sent_sims = self._cal_masked_sent_sims(sub_sents_vecs, sents_vecs) 
        sub_loss = super(SimCSELoss1, self).forward(sub_masked_sent_sims, sub_labels)

        if reduction == 'none':
            return sub_loss
        elif reduction == 'mean':
            return sub_loss.mean()
        elif reduction == 'sum':
            return sub_loss.sum()

def extended_with_Rdropout(BaseLossModule, alpha:float=4.)->nn.Module:
    '''
    Input:
    :@param BaseLossModel: class of loss module in pytorch, which is class rather than instance of the loss class
    :@param alpha: parameter for R-dropout
    :@param keys4outputLogits: 

    Details:
        - only the input for criterion is logits and labels, we could use this function to extend R-dropout
        - s_{2i} and s_{2i+1} are from the same pair of sentence
        - only support mean metrics
    Question: 
        - I am not pretty sure whether this one could work well for SBERT, cause the impact of R-dropout would directly impact on the SBERT (encoder and clf)
            - not only on encoder[but I have not set dropout before clf, so this might work]
    
    '''
    class LossWithRdropout_label(BaseLossModule, KLDivLoss):
    
        def forward(self, outputs_logits:torch.Tensor, labels: torch.Tensor):
            
            self.alpha = alpha
            self.reduction = 'mean'

            avg_CELoss = BaseLossModule.forward(self, outputs_logits, labels) # (batchSize*2, )
            
            outputs_logits_1 = outputs_logits[::2, :]   # (batchSize, n_labels)
            outputs_logits_2 = outputs_logits[1::2, :]  # (batchSize, n_labels)
            log_probs_1 = F.log_softmax(outputs_logits_1, dim=-1)   # (batchSize, n_labels)
            log_probs_2 = F.log_softmax(outputs_logits_2, dim=-1)   # (batchSize, n_labels)

            self.reduction = 'batchmean'
            self.log_target = True

            avg_KDLoss_1 = KLDivLoss.forward(self, log_probs_1, log_probs_2)    # scalar
            avg_KDLoss_2 = KLDivLoss.forward(self, log_probs_2, log_probs_1)    # scalar
            avg_KDLoss = avg_KDLoss_1 + avg_KDLoss_2    # (scalar, )
            
            loss = avg_CELoss + avg_KDLoss * self.alpha / 4 
            # L_{i} = L_{i}^{CE} + \alpha L_{i}^{KL} https://kexue.fm/archives/8496
            ## 需要除以4才可以对齐

            return loss

def extended_with_Rdropout_sent(BaseLossModule, alpha:float=4.)->nn.Module:
    '''
    Input:
    :@param BaseLossModel: class of loss module in pytorch, which is class rather than instance of the loss class
    :@param alpha: parameter for R-dropout
    :@param keys4outputLogits: 

    Details:
        - the input for criterion is logits, labels, sents_vecs1 and sents_vecs2 we could use this function to extend R-dropout
        - o_{2i} and o_{2i+1} (from output_logits)are from the same pair of sentence
        - sents_vecs1/2 is the vstacking of sents1 and sents2, which means sents_vecs1/2: (batchSize*2, hidden_nums)
        - R-dropout will calculated based on 
        - only support mean metrics
    Question: 
        - I am not pretty sure whether this one could work well for SBERT, cause the impact of R-dropout would directly impact on the SBERT (encoder and clf)
            - not only on encoder[but I have not set dropout before clf, so this might work]
    
    '''
    class LossWithRdropout_sents(BaseLossModule, KLDivLoss):
    
        def forward(self, sents_vecs1:torch.Tensor, sents_vecs2:torch.Tensor ,outputs_logits:torch.Tensor, labels: torch.Tensor, logger=None):
            
            
            self.alpha = alpha
            self.reduction = 'mean'

            avg_CELoss = BaseLossModule.forward(self, outputs_logits, labels) # (batchSize*2, )
            

            log_probs_sents_1 = F.log_softmax(sents_vecs1, dim=-1)   # (batchSize * 2, hidden_nums)
            log_probs_sents_2 = F.log_softmax(sents_vecs2, dim=-1)   # (batchSize * 2, hidden_nums)

            self.reduction = 'batchmean'
            self.log_target = True

            avg_KDLoss_1 = KLDivLoss.forward(self, log_probs_sents_1, log_probs_sents_2)    # scalar
            avg_KDLosss_2 = KLDivLoss.forward(self, log_probs_sents_2, log_probs_sents_1)    # scalar
            avg_KDLoss = 2 * (avg_KDLoss_1 + avg_KDLosss_2)    # (scalar, )
            if logger is not None:
                logger.info('')
            
            
            loss = avg_CELoss + avg_KDLoss * self.alpha / 4 
            # L_{i} = L_{i}^{CE} + \alpha L_{i}^{KL} https://kexue.fm/archives/8496
            ## 需要除以4才可以对齐

            return loss

    return LossWithRdropout_sents

def extended_with_unsup_simcse(BaseLossModule, alpha:float=4.)->nn.Module:
    '''
    Input:
    :@param BaseLossModel: class of loss module in pytorch, which is class rather than instance of the loss class
    :@param alpha: parameter for R-dropout
    :@param keys4outputLogits: 

    Details:
        - the input for criterion is logits, labels, sents_vecs1 and sents_vecs2 we could use this function to extend R-dropout
        - o_{2i} and o_{2i+1} (from output_logits)are from the same pair of sentence
        - sents_vecs1/2 is the vstacking of sents1 and sents2, which means sents_vecs1/2: (batchSize*2, hidden_nums)
        - R-dropout will calculated based on 
        - only support mean metrics
    Question: 
        - I am not pretty sure whether this one could work well for SBERT, cause the impact of R-dropout would directly impact on the SBERT (encoder and clf)
            - not only on encoder[but I have not set dropout before clf, so this might work]
    
    '''
    if issubclass(SimCSELoss, BaseLossModule) or issubclass(BaseLossModule, SimCSELoss):
        class LossWithUnsupSimCSE(SimCSELoss):

            def _convert_sents_for_simcse(self, sents_vecs1:torch.Tensor, sents_vecs2:torch.Tensor):
                '''
                - Input:
                :@param sents_vecs1/2: (batchSize * 2, hidden_nums)
                
                - Detailed:
                    - sents_vecs1 and 2 are the embedding from different dropout
                    - we need to construct the data into the way like s_{2i} and s_{2i+1} are from the same sentences
                '''
                sampleSize, hidden_nums = sents_vecs1.shape     #  sampleSize == 2 * batchSize
                combined_sents_vecs = torch.concat([sents_vecs1, sents_vecs2], dim=-1) # (2 * batchSize, 2 * hidden_nums)
                combined_sents_vecs = combined_sents_vecs.view(-1, hidden_nums) # (2 * sampleSize, hidden_nums)
                return combined_sents_vecs
        
            def forward(self, sents_vecs1:torch.Tensor, sents_vecs2:torch.Tensor ,outputs_logits:torch.Tensor, labels: torch.Tensor):
            
                self.alpha = alpha
                self.reduction = 'mean'

                # CE
                avg_CELoss = BaseLossModule.forward(self, outputs_logits, labels) # scalar

                # InfoNCE: SimCSE
                combined_sents_vecs = self._convert_sents_for_simcse(sents_vecs1, sents_vecs2) # (2 * sampleSize, hidden_nums) | # (2 * 2 * batchSize, hidden_nums)
                avg_SimCSELoss = SimCSELoss.forward(self, combined_sents_vecs, self.reduction)   # scalar
                avg_SimCSELoss = 2 * avg_SimCSELoss # 一个sample中应该是将

                loss = avg_CELoss + avg_SimCSELoss * self.alpha / 4 
                # L_{i} = L_{i}^{CE} + \alpha L_{i}^{KL} https://kexue.fm/archives/8496
                ## 需要除以4才可以对齐
                return loss
    else:
        class LossWithUnsupSimCSE(BaseLossModule, SimCSELoss):

            def _convert_sents_for_simcse(self, sents_vecs1:torch.Tensor, sents_vecs2:torch.Tensor):
                '''
                - Input:
                :@param sents_vecs1/2: (batchSize * 2, hidden_nums)
                
                - Detailed:
                    - sents_vecs1 and 2 are the embedding from different dropout
                    - we need to construct the data into the way like s_{2i} and s_{2i+1} are from the same sentences
                '''
                sampleSize, hidden_nums = sents_vecs1.shape     #  sampleSize == 2 * batchSize
                combined_sents_vecs = torch.concat([sents_vecs1, sents_vecs2], dim=-1) # (2 * batchSize, 2 * hidden_nums)
                combined_sents_vecs = combined_sents_vecs.view(-1, hidden_nums) # (2 * sampleSize, hidden_nums)
                return combined_sents_vecs
        
            def forward(self, sents_vecs1:torch.Tensor, sents_vecs2:torch.Tensor ,outputs_logits:torch.Tensor, labels: torch.Tensor):
                
                self.alpha = alpha
                self.reduction = 'mean'

                # CE
                avg_CELoss = BaseLossModule.forward(self, outputs_logits, labels) # scalar

                # InfoNCE: SimCSE
                combined_sents_vecs = self._convert_sents_for_simcse(sents_vecs1, sents_vecs2) # (2 * sampleSize, hidden_nums) | # (2 * 2 * batchSize, hidden_nums)
                avg_SimCSELoss = SimCSELoss.forward(self, combined_sents_vecs, self.reduction)   # scalar
                avg_SimCSELoss = 2 * avg_SimCSELoss # 一个sample中应该是将

                loss = avg_CELoss + avg_SimCSELoss * self.alpha / 4 
                # L_{i} = L_{i}^{CE} + \alpha L_{i}^{KL} https://kexue.fm/archives/8496
                ## 需要除以4才可以对齐
                return loss

    return LossWithUnsupSimCSE


if __name__ == '__main__':
    # criterion = SimCSELoss()
    # sents_vecs = torch.randn((6,100))
    # loss = criterion(sents_vecs, 3, reduction='none')
    
    # print(loss)

    #############################################################################

    # criterion = extended_with_Rdropout(CrossEntropyLoss)()
    # outputs_logits = torch.randn((int(5*2), 3))
    # labels = torch.randint(0, 3, (10,))
    # loss = criterion(outputs_logits, labels)

    # print(criterion.log_target, criterion.alpha, criterion.reduction)
    # print(loss)
    # print(123)

    #######################################################################

    # criterion = extended_with_Rdropout_sent(CrossEntropyLoss)()
    # sents_vecs_1 = torch.randn((5*2, 768))
    # sents_vecs_2 = torch.randn((5*2, 768))
    # outputs_logits = torch.randn((int(5*2), 3))
    # labels = torch.randint(0, 3, (10,))
    # loss = criterion(sents_vecs_1, sents_vecs_2, outputs_logits, labels) 

    # print(criterion.log_target, criterion.alpha, criterion.reduction)
    # print(loss)
    # print(123)
    
    ########################################################################

    criterion = extended_with_unsup_simcse(CrossEntropyLoss)()
    sents_vecs_1 = torch.randn((5*2, 768))
    sents_vecs_2 = torch.randn((5*2, 768))
    outputs_logits = torch.randn((int(5*2), 3))
    labels = torch.randint(0, 3, (10,))
    loss = criterion(sents_vecs_1, sents_vecs_2, outputs_logits, labels) 

    print(criterion.alpha, criterion.reduction)
    print(loss)
    print(123)