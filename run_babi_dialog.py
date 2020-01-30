from absl import flags, logging, app
from tensorboard.plugins.hparams import api as hp
import json
import os
import tensorflow as tf

from utils.data import babi_dialog
from utils import preprocessing, vocabulary
from model import build_keras_model
from utils.hparams import *

FLAGS = flags.FLAGS

# Required flags
flags.DEFINE_enum(
    'mode', None,
    ['train', 'test', 'interactive'],
    'Mode to run.')
flags.DEFINE_string(
    'data_dir', None,
    'Input data directory.')
flags.DEFINE_integer(
    'task', None,
    'bAbi task to use.')

# Optional flags
flags.DEFINE_string(
    'tb_log_dir', './logs',
    'Directory to save Tensorboard logs.')
flags.DEFINE_string(
    'model_dir', './model',
    'Directory to save model.')


def get_dataset_fn(tokenizer_fn,
                   vocab_table,
                   candidates_table,
                   hparams):

    def _dataset_fn(suffix, base_path, task):

        dataset, dataset_size = babi_dialog.load_dataset(
            suffix=suffix, 
            base_path=base_path,
            hparams=hparams, 
            task=task)

        dataset = preprocessing.preprocess_dataset(dataset,
            tokenizer_fn=tokenizer_fn,
            vocab_table=vocab_table,
            candidates_table=candidates_table,
            hparams=hparams)

        return dataset, dataset_size

    return _dataset_fn


def build_experiment_fn():

    all_texts = babi_dialog.load_all_texts(FLAGS.data_dir, 
        task=FLAGS.task)
    candidates = babi_dialog.get_candidates(FLAGS.data_dir, 
        task=FLAGS.task)
    
    def _run_experiment(num, hparams):

        run_dir = os.path.join(FLAGS.tb_log_dir, 'hparam_tuning', str(num))

        with tf.summary.create_file_writer(run_dir).as_default():

            _hparams = {h.name: hparams[h] for h in hparams}
            hparams_str = json.dumps(_hparams, indent=4)

            logging.info('Running experiment {} with hyperparameters:\n{}'.format(num, hparams_str))

            tokenizer_fn = preprocessing.get_tokenizer_fn(hparams)

            vocab, unk_id = vocabulary.build_vocab(all_texts, 
                tokenizer_fn=tokenizer_fn, hparams=hparams)

            vocab_table = preprocessing.build_lookup_table(vocab,
                default_value=unk_id)
            candidates_table = preprocessing.build_lookup_table(candidates)

            dataset_fn = get_dataset_fn(
                tokenizer_fn=tokenizer_fn,
                vocab_table=vocab_table,
                candidates_table=candidates_table,
                hparams=hparams)

            train_dataset, train_size = dataset_fn(
                suffix='trn',
                base_path=FLAGS.data_dir,
                task=FLAGS.task)
            dev_dataset, dev_size = dataset_fn(
                suffix='dev',
                base_path=FLAGS.data_dir,
                task=FLAGS.task)

            vocab_size = len(vocab)
            num_candidates = len(candidates)

            candidates_tok = tokenizer_fn(candidates).to_tensor()
            candidates_enc = vocab_table.lookup(candidates_tok)

            logging.info('Vocabulary size: {}'.format(len(vocab)))
            logging.info('# of candidates: {}'.format(len(candidates)))

            keras_model = build_keras_model(
                vocab_size=vocab_size,
                candidates=candidates_enc,
                hparams=hparams)

            optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE])

            keras_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            steps_per_epoch = train_size // hparams[HP_BATCH_SIZE]
            validation_steps = dev_size // hparams[HP_BATCH_SIZE]

            hp.hparams(hparams)

            model_dir = os.path.join(FLAGS.model_dir, str(num))
            os.makedirs(model_dir, exist_ok=True)

            py_vocab = [tok.decode('utf8') for tok in vocab.numpy()]
            vocabulary.save_vocab(py_vocab, model_dir)

            keras_model.fit(train_dataset, 
                epochs=hparams[HP_EPOCHS], 
                steps_per_epoch=steps_per_epoch,
                    callbacks=[
                        tf.keras.callbacks.TensorBoard(run_dir),
                        hp.KerasCallback(run_dir, hparams)
                    ])

            keras_model.save(model_dir)

            loss, accuracy = keras_model.evaluate(dev_dataset, 
                steps=validation_steps)

            tf.summary.scalar(METRIC_LOSS, loss, step=1)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

        return loss, accuracy

    return _run_experiment


def train():

    run_experiment_fn = build_experiment_fn()

    log_hparams_path = os.path.join(FLAGS.tb_log_dir, 'hparam_tuning')

    with tf.summary.create_file_writer(log_hparams_path).as_default():
        hp.hparams_config(
            hparams=[HP_TOKEN_TYPE, HP_VOCAB_SIZE, HP_EMBED_SIZE,
                     HP_MEMORY_SIZE, HP_MEMORY_HOPS, HP_BATCH_SIZE,
                     HP_EPOCHS, HP_LEARNING_RATE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                     hp.Metric(METRIC_LOSS, display_name='Loss')])

    session_num = 0

    for token_type in HP_TOKEN_TYPE.domain.values:
        for vocab_size in HP_VOCAB_SIZE.domain.values:
            for embedding_size in HP_EMBED_SIZE.domain.values:
                for memory_size in HP_MEMORY_SIZE.domain.values:
                    for memory_hops in HP_MEMORY_HOPS.domain.values:
                        for batch_size in HP_BATCH_SIZE.domain.values:
                            for epochs in HP_EPOCHS.domain.values:
                                for learning_rate in HP_LEARNING_RATE.domain.values:
                                    run_experiment_fn(session_num, {
                                        HP_TOKEN_TYPE: token_type,
                                        HP_MEMORY_SIZE: memory_size,
                                        HP_VOCAB_SIZE: vocab_size,
                                        HP_BATCH_SIZE: batch_size,
                                        HP_EMBED_SIZE: embedding_size,
                                        HP_MEMORY_HOPS: memory_hops,
                                        HP_LEARNING_RATE: learning_rate,
                                        HP_EPOCHS: epochs
                                    })
                                    session_num += 1


def main(_):

    if FLAGS.mode == 'train':
        train()


if __name__ == '__main__':

    flags.mark_flag_as_required('mode')
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('task')

    app.run(main)
