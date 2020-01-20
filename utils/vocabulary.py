import os
import tensorflow as tf

from .hparams import *


def build_vocab(texts, tokenizer_fn, hparams):

    tokens = tokenizer_fn(texts)
    all_tokens = tokens.flat_values

    unique_tokens, _, token_counts = tf.unique_with_counts(all_tokens)

    # Add blank for memory padding
    unique_tokens = tf.concat([[''], unique_tokens], axis=0)

    # Add T = 1000 "time features"
    time_features = ['#{}'.format(i) for i in range(1000)]
    unique_tokens = tf.concat([unique_tokens, time_features], axis=0)

    # Add speaker features
    unique_tokens = tf.concat([unique_tokens, ['<u>', '<r>']], axis=0)

    if hparams[HP_VOCAB_SIZE] <= 0:
        return unique_tokens

    # TODO: Implement fixed vocab size
    

# def save_vocab(idx_to_tok, path, prefix):

#     vocab_fp = os.path.join(path, '{}.vocab'.format(prefix))

#     with open(vocab_fp, 'w') as f:
#         for tok in idx_to_tok:
#             f.write('{}\n'.format(tok))
