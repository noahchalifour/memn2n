import tensorflow as tf

from utils.hparams import *


def build_keras_model(vocab_size,
                      candidates_enc,
                      hparams):

    memories = tf.keras.Input(shape=[hparams[HP_MEMORY_SIZE.name], None],
        name='memories', dtype=tf.int32)
    inputs = tf.keras.Input(shape=[None],
        name='inputs', dtype=tf.int32)

    A = tf.keras.layers.Embedding(vocab_size,
        output_dim=hparams[HP_EMBED_SIZE.name])
    R = tf.keras.layers.Dense(hparams[HP_EMBED_SIZE.name])
    W = tf.keras.layers.Embedding(vocab_size,
        output_dim=hparams[HP_EMBED_SIZE.name])

    mem_emb = A(memories)
    mem_emb = tf.reduce_sum(mem_emb, axis=2)

    inp_emb = A(inputs)
    inp_emb = tf.reduce_sum(inp_emb, axis=1)

    for _ in range(hparams[HP_MEMORY_HOPS.name]):

        prob = tf.matmul(inp_emb, mem_emb, 
            transpose_b=True)
        prob = tf.nn.softmax(prob)

        out = tf.matmul(prob, mem_emb)
        out = tf.reduce_sum(out, axis=1)
        out = R(out) + inp_emb

        inp_emb = out

    cand_emb = W(candidates_enc)
    cand_emb = tf.reduce_sum(cand_emb, axis=1)

    logits = tf.matmul(out, cand_emb,
        transpose_b=True)

    preds = tf.nn.softmax(logits)

    return tf.keras.Model(inputs=[memories, inputs], outputs=[preds])