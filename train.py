# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
# (https://creativecommons.org/licenses/by-nc/4.0/)

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.models.llama.modeling_llama as llama


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):
    # 학습용 시드 설정
    train_seed = opt.seed
    torch.manual_seed(train_seed)
    
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=8,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    model.train()
    while step < opt.total_steps:
        epoch += 1
        logger.info(f"Epoch {epoch} started")
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask, question_ids, question_mask, _, _) = batch
            

            context_ids = context_ids.view(opt.per_gpu_batch_size, -1) 
            context_mask = context_mask.view(opt.per_gpu_batch_size, -1)
            encoded_context = model.encode(
                input_ids = context_ids.to(model.device),
                attention_mask = context_mask.to(model.device)
            )

            model.set_cross_inputs(
                hidden_states = encoded_context[0].to(model.device),
                attention_mask = context_mask.to(model.device),
            )
            
            
            train_loss = model(
                input_ids=question_ids.to(model.device),
                attention_mask=question_mask.to(model.device),
                labels=labels.to(model.device),
            )[0]

            train_loss.backward()
            logger.info(f"step: {step}, train_loss: {train_loss}")
            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                # 평가 시에는 다른 시드 사용 (step을 시드로 활용)
                eval_seed = opt.seed + step
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt, eval_seed)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt, eval_seed):
    # 평가용 시드 설정
    torch.manual_seed(eval_seed)
    
    # 평가할 스텝 수만큼만 데이터를 사용하도록 수정
    eval_size = min(len(dataset), opt.eval_steps * opt.per_gpu_eval_batch_size)
    eval_indices = torch.randperm(len(dataset))[:eval_size]
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)
    
    sampler = SequentialSampler(eval_subset)
    dataloader = DataLoader(
        eval_subset,
        sampler=sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        num_workers=8,
        collate_fn=collator
    )
    
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, input_ids, input_mask, question_ids, question_mask) = batch

            context_ids = context_ids.view(opt.per_gpu_eval_batch_size, -1) 
            context_mask = context_mask.view(opt.per_gpu_eval_batch_size, -1)

            logger.info(f"step: {i}\n")

            encoded_context = model.encode(
                input_ids = context_ids.to(model.device),
                attention_mask = context_mask.to(model.device),
            )

            model.set_cross_inputs(
                hidden_states = encoded_context[0].to(model.device),
                attention_mask = context_mask.to(model.device),
            )

            # 질문만 입력으로 사용하고 정답은 제외
            outputs = model.generate(
                input_ids=question_ids.to(model.device),
                attention_mask=question_mask.to(model.device),
                max_length=50,
                pad_token_id=tokenizer.pad_token_id,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                logger.info(f"\n{ans}\ngold: {gold}\nscore: {score}")
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

def print_trainable_parameters(model):
    logger.info("=== Trainable Parameters ===")
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        

        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(f"Layer: {name} | Shape: {param.shape} | Trainable: Yes")
    

    logger.info("\n=== Summary ===")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("=================")

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 'meta-llama/llama-3.2-' + opt.model_size
    model_class = llama.LlamaBiCodeLM

    #load data
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = model_class.pad_token_id
    tokenizer.pad_token = tokenizer.added_tokens_decoder[tokenizer.pad_token_id].content
    tokenizer.padding_side = 'left'
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)
    
    if opt.model_path == "none":
        model = llama.LlamaBiCodeLM.from_pretrained(model_name)
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    if opt.cross_attention_layer_only == True:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.cross_layers.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    print_trainable_parameters(model)
    logger.info("Start training")

    
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )