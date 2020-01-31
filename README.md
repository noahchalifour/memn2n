# MemN2N

This repository is a Python implementation of Facebook's research paper entitled [LEARNING END-TO-END GOAL-ORIENTED DIALOG](https://arxiv.org/pdf/1605.07683.pdf) using Tensorflow 2.

## bAbi Dialog

### Training a model

To get started training a model first you must download the bAbi dialog dataset which can be found [here](https://fb-public.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w).

Once you have the dataset extracted, run the following commands to setup the repository:

```
git clone https://github.com/noahchalifour/memn2n
cd memn2n
pip install tf-nightly==2.2.0.dev20200130 # or tf-nightly-gpu==2.2.0.dev20200130 for GPU version
pip install -r requirements.txt
```

Once you have the code setup, you can modify the model hyperparameters in the `utils/hparams.py` file. By default the hyperparameters are setup to run a bunch of test so if you just want to train a single model, set all hyperparameter values to a list of the single value you want to use.

After you've modified the hyperparameters (Optional), run the following command to start training:

```
python run_babi_dialog.py \
    --mode train \
    --task {{ task_id }} \
    --data_dir {{ babi_dialog_dir }}
```

### Testing your model

Once you have a model trained, you can test your model by using the following command:

```
python run_babi_dialog.py \
    --mode test \
    --task {{ task_id }} \
    --data_dir {{ babi_dialog_dir }} \
    --model_dir {{ model_dir }} \
    --use_oov {{ true to test on OOV, false otherwise }}
```

### Results

All of the following models were trained for 200 epochs with an `embedding_size` = 32, `memory_size` = 50, `memory_hops` = 3, `learning_rate` = 1e-03, and `batch_size` = 32. Better results can be achieved by tuning hyperparameters. (most notably embedding_size and memory_hops)

Task | Original Paper | This Repository
--- | --- | ---
T1: Issuing API calls | **99.9** | **99.9**
T2: Updating API calls | **100** | 99.9
T3: Displaying options | **74.9** | 74.6
T4: Providing information | **59.5** | 56.7
T5: Full dialogs | **96.1** | 92.6
T1 (OOV): Issuing API calls | 72.3 | **81.9**
T2 (OOV): Updating API calls | **78.9** | 78.8
T3 (OOV): Displaying options | **74.4** | 69.2
T4 (OOV): Providing information | **57.6** | 57.1
T5 (OOV): Full dialogs | **65.5** | 63.0
T6: Dialog state tracking 2 | **41.1** | 39.2
