from tensorboard.plugins.hparams import api as hp

HP_TOKEN_TYPE = hp.HParam('token_type', hp.Discrete(['word']))
HP_VOCAB_SIZE = hp.HParam('vocab_size', hp.Discrete([-1]))
HP_EMBED_SIZE = hp.HParam('embedding_size', hp.Discrete([32, 128]))

HP_MEMORY_SIZE = hp.HParam('memory_size', hp.Discrete([50, 100, 250]))
HP_MEMORY_HOPS = hp.HParam('memory_hops', hp.Discrete([1, 3, 5, 10]))

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([200]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4]))

METRIC_LOSS = 'loss'
METRIC_ACCURACY = 'accuracy'