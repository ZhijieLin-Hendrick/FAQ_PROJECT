from .train import train4simcse, train4sbert
from .utils import util
from transformers import BertConfig, BertTokenizer, BertModel
from .model_module.model import SBERT, SRoformer
import torch



def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "WangZeJun/simbert-base-chinese"
    encoder_model = 'simbert'

    for batch_size in [16, 32, 64, 128]:
        print(f"##############################batchSize: {batch_size}################################")
        logger_file_name = f'./FAQ project/log/sup_sbert_rdropoutSent_{encoder_model}_batchSize{batch_size}_earlyStops50k.log'
        saveModel_path = f'./FAQ project/tmp_data/model_checkpoint/supSBERT_RDropoutSent_{encoder_model}_batchSize{batch_size}_earlyStops50k.ckpt'
        saveTestMetrics_path = f'./FAQ project/tmp_data/supSBERT_{encoder_model}_batchSize{batch_size}_earlyStops50k_test'

        logger = util.get_logger(logger_file_name)

        config = BertConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = 0.3
        config.n_labels = 2
        simbert_model = SBERT(config, model_name)
        simbert_model.to(device)
        tokz = BertTokenizer.from_pretrained(model_name)
        logger.info('successfully load model and tokenizer')

        trainer = train4sbert.Trainer4SupBERT_RDropout_Sent(
            train_path='./FAQ project/data/merged_qq_train.csv',
            dev_path='./FAQ project/data/merged_qq_dev.csv',                 
            test_path='./FAQ project/data/merged_qq_test.csv', 
            device=device,lr=1e-5, tokz=tokz, batchSize=16, 
            dev_batchSize=5000, model=simbert_model, max_epoch=30, 
            logger=logger, eval_steps=30, save_path=saveModel_path,
            early_stop=5e4
        )

        trainer.fit()

        # evaluate on test_set
        model = SBERT.load(saveModel_path, device)
        model.eval()
        test_iter = util.get_paired_dataIter('./FAQ project/data/merged_qq_test.csv', tokz, device, 60, 200)
        test_metrics = test(simbert_model.encoder, test_iter, f'./FAQ project/tmp_data/supSBERT_{encoder_model}_batchSize{batch_size}_earlyStops50k_test')
        logger.info(f"Test, test_auc: {test_metrics['auc']}, test_spearman: {test_metrics['spearman']}")


if __name__ == '__main__':
    main()