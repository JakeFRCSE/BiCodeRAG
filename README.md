## Research Documentation
Please click the link to access the Notion page, which is used for organizing ideas.

[Click Here!](https://crystal-air-942.notion.site/CrossDecoder-Training-Additional-Cross-Attention-Layer-in-Decoder-Only-Models-19941c6bef1680208d9af3e4f577aa8d?pvs=4)

## How to set the environment
1. Create conda virtual environment
```
conda create -n name_env python=3.12.9
```
2. Activate conda virtual environment
```
(conda init)
conda activate name_env
```
3. Download PyTorch library
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
4. Download transformers library
```
conda install conda-forge::transformers==4.48.3
```

# How to use
1. Login into huggingface to get access to the model (this process is required only for the initialization of the model)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
2. Move to the Directory of this repository.
```bash
cd YOUR_PATH_TO_BiCodeRAG
```
3. Train
Make sure you have more than 16GM of vRAM for meta-llama/llama-3.2-1B model.

This Example is for training and evaluating on NQ dataset.

```bash
python train.py --train_data ./open_domain_data/NQ/train.json \
                --eval_data ./open_domain_data/NQ/dev.json \
                --model_size 1B \
                --per_gpu_batch_size 2 \
                --n_context 100 \
                --eval_freq 5000 \
                --warmup_steps 8000 \
                --total_steps 320000 \
                --per_gpu_eval_batch_size 1 \
                --name my_experiment \
                --checkpoint_dir checkpoint \
                --cross_attention_layer_only \
                # If you have a model to further train, add the following line.
                #--model_path YOUR_DIRECTORY_TO_MODEL \
```
4. Evaluate
```bash
python test.py \
        --model_path --model_path YOUR_DIRECTORY_TO_MODEL \
        --eval_data ./open_domain_data/NQ/test.json\
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
```


All of the files are based on [facebookresearch/FiD](https://github.com/facebookresearch/FiD/tree/main) and [transformers library](https://github.com/huggingface/transformers) 
