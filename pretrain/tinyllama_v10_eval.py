import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
from tinyllama_v9 import create_dataloaders

import os.path as osp
import csv

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters=None) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    last_targets = None

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        # print("config.block_size", model.config.block_size)
        input_ids = val_data[:, 0 : -1].contiguous()
        targets = val_data[:, 1 : ].contiguous()


        reset_hiddens = True
        # if last_targets is not None:
        #     # Compute the norm of the difference
        #     diff_norm = torch.max(torch.abs(last_targets - input_ids[:, :1]))
        #     if diff_norm > 0:
        #         print("reset hiddens")
        #         # assert False, "reset hiddens" # Find the first switch
        #         # exit()
        #         reset_hiddens = True
        #     else:
        #         reset_hiddens = False


        # input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        # targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        # print("input_ids", input_ids.shape, "targets", targets.shape)
        # logits = model(input_ids)
        logits = model(input_ids, reset_hiddens=reset_hiddens)
        # print("logits", logits.shape, "targets", targets.shape)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()
        
        last_targets = targets[:, -1:].contiguous()

    out = losses.mean()

    model.train()
    return out


evaluation_val = []
evaluation_test = []
evaluation_train = []

seq_len_list = [16 * 2 ** i for i in range(0,12)] # 16 to 32768. 
# seq_len_list = [16 * 2 ** i for i in range(7,12)] # 2048 to 32768. 
for seq_len in seq_len_list:
    
    eval_block_size = seq_len

    model_name = "tiny_Mamba_120m"
    name = "tiny_Mamba_120m_z_slow"
    out_dir = Path("/home/aiops/wangsd/TinyLlama_final/TinyLlama/out") / name
    resume =  Path("/home/aiops/wangsd/TinyLlama_final/TinyLlama/out/tiny_Mamba_120m_z_slow/iter-089600-ckpt-v10.pth")
    version = 10

    # Hyperparameters
    num_of_devices = 1
    global_batch_size = 256
    learning_rate = 6e-4
    micro_batch_size = 1
    max_step = 4800
    warmup_steps = 48
    log_step_interval = 4
    eval_iters = 64 * (seq_len_list[-1] // seq_len) # Evaluation in the same token setting 
    save_step_interval = 400
    eval_step_interval = 400


    weight_decay = 1e-1
    # weight_decay = 0.000
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    min_lr = 1e-5

    batch_size = global_batch_size // num_of_devices
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0
    warmup_iters = warmup_steps * gradient_accumulation_steps




    max_iters = max_step * gradient_accumulation_steps
    lr_decay_iters = max_iters
    log_iter_interval = log_step_interval * gradient_accumulation_steps


    # Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.


    train_data_config = [
        ("train_ind", 1.0),
    ]

    val_data_config = [
        ("test_ind", 1.0),
    ]

    test_data_config = [
        ("valid_ind", 1.0),
    ]


    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
    # wandb_logger = WandbLogger()





    devices = 1
    strategy = "auto"

    if seq_len < seq_len_list[-1]:
        precision = "32-true"
    else:
        precision = "16-true"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[])
    # fabric.print(hparams)

    config = Config.from_name(model_name)

    train_data_dir = Path("/home/aiops/wangsd/TinyLlama/data/the_pile_deduplicated_EleutherAI_split_combined")
    val_data_dir = Path("/home/aiops/wangsd/TinyLlama/data/the_pile_deduplicated_EleutherAI_split_combined")
    test_data_dir = Path("/home/aiops/wangsd/TinyLlama/data/the_pile_deduplicated_EleutherAI_split_combined")

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        # block_size=config.block_size,
        block_size=eval_block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        test_data_dir=test_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    # fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))




    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    model = state["model"]

    print("eval_iters", eval_iters)

    if val_dataloader is not None:
        t0 = time.perf_counter()
        val_loss = validate(fabric, model, val_dataloader, eval_iters=eval_iters)  # sanity check
        t1 = time.perf_counter() - t0
        fabric.print(f"Sanity check, val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")

    if test_dataloader is not None:
        t0 = time.perf_counter()
        test_loss = validate(fabric, model, test_dataloader, eval_iters=eval_iters)  # sanity check
        t1 = time.perf_counter() - t0
        fabric.print(f"Sanity check, test loss {test_loss:.4f}, test time: {t1 * 1000:.2f}ms")

    if train_dataloader is not None:
        t0 = time.perf_counter()
        train_loss = validate(fabric, model, train_dataloader, eval_iters=eval_iters)  # sanity check
        t1 = time.perf_counter() - t0
        fabric.print(f"Sanity check, train loss {train_loss:.4f}, train time: {t1 * 1000:.2f}ms")

    evaluation_val.append((seq_len, 'val', val_loss))
    evaluation_test.append((seq_len, 'test', test_loss))
    evaluation_train.append((seq_len, 'train', train_loss))

    evaluation = evaluation_val + evaluation_test + evaluation_train
    print(evaluation)

    # Write the validation losses to a CSV file
    with open(osp.join(out_dir, f'evaluation_consecutive_True.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence Length', 'Dataset Type', 'Loss'])
        writer.writerows(evaluation)



