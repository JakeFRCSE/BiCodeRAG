# Current Branch is dev/llama!
To return to the main branch, [click here!](https://github.com/JakeFRCSE/BiCodeRAG)

To view weekly research progress report (which is in korean), [click here!](https://crystal-air-942.notion.site/1a041c6bef1680e68685f7890655201b)

# How to use
1. Login into huggingface to get access to the model (this will be deprecated soon)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
2. Move to the Directory of this repository.
```bash
cd YOUR_PATH/BiCodeRAG
```
3. Train
Make sure you have more than 16GM of vRAM for meta-llama/llama-3.2-1B model.
```bash
python train.py \
        --train_data ./open_domain_data/NQ/train.json \
        --eval_data ./open_domain_data/NQ/dev.json \
        --model_size 1B \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
```

All of the files introduced at current branch is copied from [facebookresearch/FiD](https://github.com/facebookresearch/FiD/tree/main) 
