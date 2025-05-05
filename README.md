# Current Branch is dev/llama!
To return to the main branch, [click here!](https://github.com/JakeFRCSE/BiCodeRAG)

To view weekly research progress report, [click here!](https://crystal-air-942.notion.site/1a041c6bef1680e68685f7890655201b)

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
Make sure you have more than 32GM of vRAM for meta-llama/llama-3.2-1B model.

```bash
python train.py --train_data ./open_domain_data/NQ/train.json \
                --eval_data ./open_domain_data/NQ/dev.json \
                --model_size 1B \
                --per_gpu_batch_size 64 \
                --n_context 5 \
                --eval_freq 500 \
                --name NQ_1B_experiment \
                --checkpoint_dir NQ_1B_checkpoint \
                --warmup_steps 1000 \
                --total_steps 10000 \
                --per_gpu_eval_batch_size 64 \
                --cross_attention_layer_only \
                --save_freq 1237
                # If you have a model to further train, add the following line.
                #--model_path YOUR_DIRECTORY_TO_MODEL \
```
4. Evaluate
```bash
python test.py \
        --model_path --model_path YOUR_DIRECTORY_TO_MODEL \
        --eval_data ./open_domain_data/NQ/test.json\
        --per_gpu_batch_size 2 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
```
