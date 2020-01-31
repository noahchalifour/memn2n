import os
import glob
import re
import tensorflow as tf

from ..hparams import *
from ..preprocessing import preprocess_dataset, get_tokenizer_fn


def get_candidates(base_path, task):

    def clean_candidate(cand):
        _cand = re.sub(r'^\d+\s+', '', cand)
        _cand = _cand.strip('\n')
        _cand = _cand.strip()
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

    texts = []

    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                _line = line.strip('\n')
                if _line == '':
                    continue
                _line = re.sub(r'^\d+\s+', '', _line)
                line_texts = [t.strip() for t in _line.split('\t')]
                if len(line_texts) > 1:
                    texts += line_texts

    return texts


def load_kb(base_path, task):

    if task == 6:
        kb_path = os.path.join(base_path, 'dialog-babi-task6-dstc2-kb.txt')
    else:
        kb_path = os.path.join(base_path, 'dialog-babi-kb-all.txt')

    kb = []

    with open(kb_path, 'r') as f:
        for line in f:
            _line = line.strip('\n')
            if _line == '':
                continue
            _line = re.sub(r'^\d+\s+', '', _line)
            if task == 6:
                result, _type, word = _line.split(' ')
            else:
                result, word = _line.split('\t')
                result, _type = result.split(' ')
            kb.append([result, _type, word])

    return kb


def load_dataset(suffix, 
                 base_path,
                 hparams,
                 task=None):

    initial_memory = ['' for _ in range(hparams[HP_MEMORY_SIZE.name])]
    
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

                _line = re.sub(r'^\d+\s+', '', _line)

                texts = _line.split('\t')

                if len(texts) == 1:

                    inp = '#{} <u> {}'.format(conv_index, texts[0])

                    if len(memories) > 0 and conv_index != 0:
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
                    
                    inp_text = inp_text.strip()
                    out_text = out_text.strip()

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