# Tackling Low-Resource ECG Classification with Self-Supervised Learning

This repository contains implementation of Self-Supervised Learinig (SSL) models applied to mobile health (mHealth).

> Note: We adapted to our code both [TS2Vec](https://github.com/zhihanyue/ts2vec) and [TS-TCC](https://github.com/emadeldeen24/TS-TCC/tree/main) official implmentation.


## Requirements

The main requirements for this project are specified as follows:
- python=3.12
- pytorch
- pytorch-cuda=12.1
- plotly
- jupyter
- pandas
- scipy

The dependencies can be installed by:
```bash
conda env create -f environment.yml
```

## Preparing datasets
The data should be in a separate folder called 'datasets' inside the project folder. We provide notebooks to make the pre-processing of the datasets:
- **PTB-XL-700** can be analyzed and pre-processed using [ptb-xl.ipynb](./notebooks/ptb-xl.ipynb)
- **ECG Fragment** can be analyzed and pre-processed using [ecg-fragment.ipynb](./notebooks/ecg-fragment.ipynb)


## Training the models

The code also allows setting a name for the experiment, and during the folder's creation, a sufix with the timestamp will be added. Additionally, it allows the choice of a random seed value. The configuration and training strategy of each model is passed separaterly by a JSON file. You can select one of several training modes for the task configuration file.

To use these options:
```bash
python main.py <name> --upsampler_conf <path> --pretrain_conf <path> --task_conf <path> --seed 123
```


You can select one of several training modes for the task configuration file (`--task_conf`):
- Random Initialization (`"random"`)
- Supervised training (`"supervised"`)
- Fine-tuning the SSL encoder and the upsampler model (`"fine-tuning"`)
- Training only the classifier (`"linear-probing"`)

Once `--pretrain_conf` (self-supervised model) and `--upsampler_conf` (supervised model) only support one training strategy, setting then in the configuration file is optional.

**Some attentions to take**
- The name of the dataset in the configuration file should be the same as the name inside the 'datasets' folder.
- To train a classifier model you have to run the training of the SSL model first.

Exemple of configuration files are available in [conf/](./conf/).

# References

[1] Yue, Zhihan, et al. "[Ts2vec: Towards universal representation of time series](https://arxiv.org/abs/2106.10466)". Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.

[2] Eldele, Emadeldeen, et al. "[Time-series representation learning via temporal and contextual contrasting](https://arxiv.org/pdf/2106.14112.pdf)". arXiv preprint arXiv:2106.14112 (2021).