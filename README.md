Diffusion-based Generative Speech Source Separation
===================================================

This repository contains the code based on the model from [Diffusion-based Generative Speech
Source Separation](https://arxiv.org/abs/2210.17327) presented at ICASSP 2023.

The modified version conditions the existing DiffSep network with a transformer-based stage 1 network. 


Configuration
-------------

Configuration is done using the [hydra](https://hydra.cc/docs/intro/) hierarchical configuration package.
The hierarchy is as follows.
```
config/
|-- config.yaml  # main config file
|-- datamodule  # config of dataset and dataloaders
|   |-- default.yaml
|   `-- diffuse.yaml  # smaller batch size for CDiffuse
|-- model
|   |-- default.yaml  # NCSN++ model
|   `-- diffuse.yaml  # CDiffuse model
`-- trainer
    `-- default.yaml  # config of pytorch-lightning trainer
```

Dataset
-------

The `wsj0_mix` dataset is expected in `data/wsj0_mix`
```
data/wsj0_mix/
|-- 2speakers
|   |-- wav16k
|   |   |-- max
|   |   |   |-- cv
|   |   |   |-- tr
|   |   |   `-- tt
|   |   `-- min
|   |       |-- cv
|   |       |-- tr
|   |       `-- tt
|   `-- wav8k
|       |-- max
|       |   |-- cv
|       |   |-- tr
|       |   `-- tt
|       `-- min
|           |-- cv
|           |-- tr
|           `-- tt
`-- 3speakers
    |-- wav16k
    |   `-- max
    |       |-- cv
    |       |-- tr
    |       `-- tt
    `-- wav8k
        `-- max
            |-- cv
            |-- tr
            `-- tt
```

Training
--------

Preparation
```bash
conda env create -f environment.yaml
conda activate diff-sep
```
Run training. The results of training and tensorboard files are stored in `./exp/`.
```bash
python ./train.py
```
Thanks to hydra, parameters can be added easily
```bash
python ./train.py model.sde.sigma_min=0.1
```

The training can be run in **multi-gpu** setting by overriding the trainer config
`trainer=allgpus`.  Since validation is quite expensive to do, we set
`trainer.check_val_every_n_epoch=5` to run it only every 5 epochs.
The train and validation batch sizes are multiplied by the number of GPUS.

Evaluation
----------

The `evaluation.py` script can be used to run the inference for `val` and `test` datasets.
```bash
$ python ./evaluate.py --help
usage: evaluate.py [-h] [-d DEVICE] [-l LIMIT] [--save-n SAVE_N] [--val] [--test] [-N N] [--snr SNR] [--corrector-steps CORRECTOR_STEPS] [--denoise DENOISE] ckpt

Run evaluation on validation or test dataset

positional arguments:
  ckpt                  Path to checkpoint to use

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device to use (default: cuda:0)
  -l LIMIT, --limit LIMIT
                        Limit the number of samples to process
  --save-n SAVE_N       Save a limited number of output samples
  --val                 Run on validation dataset
  --test                Run on test dataset
  -N N                  Number of steps
  --snr SNR             Step size of corrector
  --corrector-steps CORRECTOR_STEPS
                        Number of corrector steps
  --denoise DENOISE     Use denoising in solver
  --enhance             Run evaluation for speech enhancement task (default: false)
```
This will save the results in a folder named `results/{exp_name}_{ckpt_name}_{infer_params}`.
The option `--save-n N` allows to save the firs `N` samples as figures and audio samples.

Reproduce
---------

### Separation

```shell
# train
python ./train.py experiment=icassp-separation

# evaluate
python ./evaluate_mp.py exp/default/<YYYY-MM-DD_hh-mm-ss>_experiment-icassp-separation/checkpoints/epoch-<NNN>_si_sdr-<F.FFF>.ckpt --split test libri-clean
```

### Enhancement

```shell
# train
python ./train.py experiment=noise-reduction

# evaluate
python ./evaluate.py exp/enhancement/<YYYY-MM-DD_hh-mm-ss>_experiment-noise-reduction/checkpoints/epoch-<NNN>_si_sdr-<F.FFF>.ckpt --test --pesq-mode wb
```

License
-------

2023 (c) LINE Corporation

The repo is released under MIT license, but please refer to individual files for their specific license.
