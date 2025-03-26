# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
# (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.models.llama.modeling_llama as llama

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, input_ids, input_mask, question_ids, question_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            # 컨텍스트 인코딩
            context_ids = context_ids.view(opt.per_gpu_batch_size, -1)
            context_mask = context_mask.view(opt.per_gpu_batch_size, -1)
            encoded_context = model.encode(
                input_ids=context_ids.to(model.device),
                attention_mask=context_mask.to(model.device)
            )

            # 크로스 어텐션 설정
            model.set_cross_inputs(
                hidden_states=encoded_context[0].to(model.device),
                attention_mask=context_mask.to(model.device),
            )

            # 생성
            outputs = model.generate(
                input_ids=question_ids.to(model.device),
                attention_mask=question_mask.to(model.device),
                max_length=50,
                pad_token_id=tokenizer.pad_token_id,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.to(model.device))

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.get_example(idx[k])
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()
                logger.info(f"\n{ans}\ngold: {example['answers']}\nscore: {score}")
                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return score, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    # Llama 모델과 토크나이저 설정
    model_name = 'meta-llama/llama-3.2-' + opt.model_size
    model_class = llama.LlamaBiCodeLM
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = model_class.pad_token_id
    tokenizer.pad_token = tokenizer.added_tokens_decoder[tokenizer.pad_token_id].content
    tokenizer.padding_side = 'left'

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank,
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=8, 
        collate_fn=collator_function
    )
    
    # 모델 로드
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.local_rank)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)
