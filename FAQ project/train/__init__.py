from pytz import NonExistentTimeError
from ..utils import util


class BaseTrainer:
    def __init__(
        self, 
        model=None,
        tokz=None,
        max_epoch:int=1,
        batch_split:int=1,  # used for simCSE
        dev_batch_split:int=1,  # used for simCSE
        batch_accumulated:int=1,    # used for gradient accumulated
        warmup_steps:int=1000,
        train_path:str=None,
        dev_path:str=None,
        batchSize:int=None,
        dev_batchSize:int=None,
        lr:float=1e-5, 
        dropout_rate:float=0.3,
        maxSeqLen:int=60,
        device=None,
        eval_steps:int=None,
        logger=None,
        save_path:str=None,
        early_stop:int=10,
        *args, **kwargs
    ):
        # 1. model
        # 2. train_iter and dev_iter
        # 3. optimizer, lr_scheduler
        # 4. loss 
        
        self.model = model
        if device is not None:
            self.device = device
        else:
            self.device = model.device
        self.tokz = tokz
        self.batchSize = batchSize
        self.MAX_EPOCH = max_epoch
        self.BATCH_SPLIT = batch_split
        if dev_batchSize is None:
            self.dev_batchSize = self.BATCH_SPLIT
        else:
            self.dev_batchSize = dev_batchSize
        self.BATCH_ACCUMULATED = batch_accumulated  # used for gradient accumulated
        self.WARMUP_STEPS = warmup_steps
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.eval_steps = eval_steps
        self.train_iter = util.get_paired_dataIter(train_path, tokz, self.device, maxSeqLen, batchSize)
        self.dev_iter = util.get_paired_dataIter(dev_path, tokz, self.device, maxSeqLen, self.dev_batchSize)
        self.criterion = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_lr_scheduler()
        self.logger = logger
        self.save_path = save_path
        self.early_stop = early_stop

        self.logger.info('Initialize successfully!')
        
        


    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def _train_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def _eval(self, *args, **kwargs):
        raise NotImplementedError

    def _get_loss(self, *args, **kwargs):
        raise NonExistentTimeError

    def _get_optimizer(self, *args, **kwargs):
        raise NonExistentTimeError

    def _get_lr_scheduler(self, *args, **kwargs):
        return None

    def _save_model(self, *args, **kwargs):
        raise NotImplementedError

    def _load_model(self, *args, **kwargs):
        raise NotImplementedError