# MemN2N

This repository is a Python implementation of Facebook's research paper entitled [LEARNING END-TO-END GOAL-ORIENTED DIALOG](https://arxiv.org/pdf/1605.07683.pdf) using Tensorflow 2.

## Usage

To get started training a model first you must download the bAbi dialog dataset which can be found [here](https://fb-public.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w).

Once you have the dataset extracted, run the following commands to setup the repository:

```
git clone https://github.com/noahchalifour/memn2n
cd memn2n
pip install tensorflow # or tensorflow-gpu for GPU version
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