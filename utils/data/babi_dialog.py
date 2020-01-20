import os
import glob
import re
import tensorflow as tf

from ..hparams import *
from ..preprocessing import preprocess_dataset, get_tokenizer_fn


# def get_dataset_fn(input_file_pattern):

#     dataset = tf.data.Dataset.list_files(input_file_pattern)

#     dataset = dataset.interleave(
#       tf.data.TextLineDataset, cycle_length=4,
#       num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     for d in dataset.take(1):
#         print(d)


# def load_dataset(ds_type, path, hparams, task=None):

#     if task is None:
#         input_file_pattern = os.path.join(path, 
#             'dialog-babi-task*-{}.txt'.format(ds_type))
#     else:
#         input_file_pattern = os.path.join(path,
#             'dialog-babi-task{}-*-{}.txt'.format(task, ds_type))

#     return get_dataset_fn(input_file_pattern)


def get_candidates(base_path, task):

    def clean_candidate(cand):
        _cand = re.sub(r'^\d+\s', '', cand)
        _cand = _cand.strip('\n')
        return _cand

    if task == 6:
        candidates_fn = 'dialog-babi-task6-dstc2-candidates.txt'
    else:
        candidates_fn = 'dialog-babi-candidates.txt'

    candidates_fp = os.path.join(base_path, candidates_fn)

    with open(candidates_fp, 'r') as f:
        lines = f.readlines()

    all_candidates = [clean_candidate(c) for c in lines]

    return all_candidates


def load_all_texts(base_path, task):

    if task == 6:
        filepaths = glob.glob(os.path.join(base_path,
            'dialog-babi-task6*.txt'))
    else:
        filepaths = glob.glob(os.path.join(base_path,
            'dialog-babi-task[1-5]-*[s|l|v|n|t].txt'))
        filepaths += [
            os.path.join(base_path, 'dialog-babi-candidates.txt'),
            os.path.join(base_path, 'dialog-babi-kb-all.txt')]

    texts = []

    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                _line = line.strip('\n')
                if _line == '':
                    continue
                _line = re.sub(r'^\d+\s', '', _line)
                texts += _line.split('\t')

    return texts


def load_dataset(suffix, 
                 base_path,
                 hparams,
                 task=None):

    initial_memory = ['' for _ in range(hparams[HP_MEMORY_SIZE])]
    
    memories = []
    inputs = []
    outputs = []
    
    if task is None:
        glob_pattern = os.path.join(base_path, 
            'dialog-babi-task*-{}.txt'.format(suffix))
    else:
        glob_pattern = os.path.join(base_path, 
            'dialog-babi-task{}-*-{}.txt'.format(task, suffix))

    is_new_dialog = True
    is_knowledge = False
    conv_index = 0

    for filepath in glob.glob(glob_pattern):

        with open(filepath, 'r') as f:
            
            for line in f:

                _line = line.strip('\n')

                if _line == '':
                    conv_index = 0
                    is_new_dialog = True
                    continue

                _line = re.sub(r'^\d+\s', '', _line)

                texts = _line.split('\t')

                if len(texts) == 1:

                    inp = '#{} <u> {}'.format(conv_index, texts[0])

                    if len(memories) > 0:
                        new_memory = memories[-1][1:] + [inp]
                    else:
                        new_memory = initial_memory[1:] + [inp]

                    if is_knowledge and len(memories) > 0:
                        memories[-1] = new_memory
                    else:
                        is_knowledge = True
                        memories.append(new_memory)

                    conv_index += 1

                else:
                
                    inp_text, out_text = texts

                    if is_knowledge:

                        is_knowledge = False
                        inputs.append(inp_text)
                        outputs.append(out_text)

                    else:

                        if is_new_dialog:
                            new_memory = initial_memory
                            is_new_dialog = False
                        else:
                            last_memory = memories[-1]
                            mem_inp = '#{} <u> {}'.format(conv_index, inputs[-1])
                            mem_out = '#{} <r> {}'.format(conv_index + 1, outputs[-1])
                            new_memory = last_memory[2:] + [mem_inp, mem_out]
                            conv_index += 2

                        memories.append(new_memory)
                        inputs.append(inp_text)
                        outputs.append(out_text)

                is_new_dialog = False

    dataset = tf.data.Dataset.from_tensor_slices({
        'memories': memories,
        'inputs': inputs, 
        'outputs': outputs
    })

    size = len(inputs)

    return dataset, size