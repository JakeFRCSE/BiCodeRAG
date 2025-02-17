## Currently Working on LLaMA
Please switch to the [feature/llama](https://github.com/JakeFRCSE/CrossDecoder/tree/feature/llama) branch to view the latest progress.

## Research Documentation
Please click the link to access the Notion page, which is used for organizing ideas.

[Click Here!](https://crystal-air-942.notion.site/CrossDecoder-Training-Additional-Cross-Attention-Layer-in-Decoder-Only-Models-19941c6bef1680208d9af3e4f577aa8d?pvs=4)

## How to set the environment by conda_requirements.txt
1. download conda_requirements.txt
  
2. Run the following code (Anaconda3 must be already installed)
```
conda env create -f conda_requirements.txt
```

## How to set the environment from scratch
1. Create conda virtual environment
```
conda create -n cross_decoder_env python=3.12.9
```
2. Activate conda virtual environment
```
(conda init)
conda activate name_env
```
3. Download transformers library
```
conda install conda-forge::transformers==4.48.3
```
4. Download PyTorch library
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
