import time
import os
from pathlib import Path
from typing import Optional, Dict

import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pnlp

from ..utils import get_lora_state_dict, get_date_of_run


class Trainer:

    def __init__(
        self,
        lr: float,
        num_epochs: int,
        warmup_steps: int,
        accumulate_steps: int,
        out_dir: str,
        device: Optional[str],
        print_every: Optional[int],
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.accumulate_steps = accumulate_steps
        self.out_dir = out_dir
        self.device = device
        self.print_every = print_every
        self.max_to_keep = 5

    def train(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        is_distributed: bool = False,
        rank: int = 0,
    ) -> None:
        if rank == 0:
            self.ckpt_path = os.path.join(self.out_dir, "ckpt")
            pnlp.check_dir(self.out_dir, self.ckpt_path)
            time_of_run = get_date_of_run()
            self.save_file_prefix = f"lora-{time_of_run}-ckpt-best"
            start_time = time.perf_counter()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        if self.accumulate_steps is None:
            accumulate_steps = 1
        else:
            accumulate_steps = self.accumulate_steps
        batch_num = len(train_dataloader)
        epoch_update_steps = int(batch_num / accumulate_steps)
        if self.warmup_steps is None:
            warmup_update_steps = epoch_update_steps
        else:
            warmup_update_steps = self.warmup_steps
        total_update_steps = epoch_update_steps * self.num_epochs

        # stops when 3 epochs do not improve
        early_stop_steps = 3 * batch_num
        # every 1/3 epoch do evaluation
        valid_steps = batch_num // 3 + 1

        # every 1/10 epoch print
        if self.print_every is None:
            print_every = batch_num // 10
        else:
            print_every = self.print_every
        print_every = max(print_every, 1)

        if rank == 0:
            msg = f"Total data batches: {batch_num}, "
            msg += f"Epoch update steps: {epoch_update_steps}, "
            msg += f"Total update steps: {total_update_steps}, "
            msg += f"\nWarmup update steps: {warmup_update_steps}, "
            msg += f"Validation steps: {valid_steps}, "
            msg += f"Early stop steps: {early_stop_steps}, "
            msg += f"Print every: {print_every}"
            print(msg)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_update_steps,
            num_training_steps=total_update_steps,
        )

        val_best_loss = float("inf")
        total_step = 0
        last_improve = 0
        flag = False
        for epoch in range(self.num_epochs):
            model.train()
            if is_distributed:
                train_loss = torch.zeros(1).to(rank)
                train_dataloader.sampler.set_epoch(epoch)
            else:
                train_loss = torch.zeros(1).to(self.device)
            if rank == 0:
                print(f"\n\nEpoch {epoch}/{self.num_epochs}")
                inner_pbar = tqdm.tqdm(
                    range(len(train_dataloader)), colour="blue", 
                    desc="rank0 Training Epoch"
                )
            for step, batch in enumerate(train_dataloader, start=1):
                if is_distributed:
                    output = self._fsdp_step(model, batch, rank)
                else:
                    output = self._step(model, batch)
                
                # the loss has already been averaged along batch
                loss_b = output.loss.detach().float()
                loss = output.loss / accumulate_steps
                loss.backward()
                if step % accumulate_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                train_loss[0] += loss_b
                total_step += 1
                if rank == 0:
                    inner_pbar.update(1)

                if rank == 0 and step % print_every == 0:
                    msg = f"Step: {step}, "
                    msg += f"Loss: {train_loss[0] / step:.4f}, "
                    msg += f"LearningRate: {lr_scheduler.get_last_lr()[0]:.6f} "
                    print(msg)

                if (
                    total_step % valid_steps == 0 and 
                    valid_steps < batch_num
                ):
                    val_loss = self.eval(model, dev_dataloader, is_distributed, rank)
                    if val_loss < val_best_loss:
                        self._save(model, total_step, is_distributed, rank)
                        val_best_loss = val_loss
                        last_improve = total_step

                    if rank == 0:
                        msg = "- " * 30 + "\n"
                        msg += f"Step(Batch)/Epoch: {step}/{epoch},  Total Step: {total_step},  "
                        msg += f"LearningRate: {lr_scheduler.get_last_lr()[0]:.6f}, \n"
                        msg += f"TrainLoss: {train_loss[0] / step:.4f},  "
                        msg += f"ValidLoss: {val_loss:.4f}\n"
                        msg += "= " * 30
                        print(msg)
                    
                    model.train()

                if rank == 0 and total_step - last_improve > early_stop_steps:
                    print("Early stop for no improvements...")
                    flag = True

            if is_distributed:
                dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                inner_pbar.close()
            
            val_loss = self.eval(model, dev_dataloader, is_distributed, rank)
            if val_loss < val_best_loss:
                self._save(model, total_step, is_distributed, rank)
                val_best_loss = val_loss
                last_improve = total_step
            
            ##### Summary Epoch #####
            if rank == 0:
                secs = int(time.perf_counter() - start_time)
                mins = secs / 60
                secs = secs % 60
                msg = f"Epoch: {epoch} | time in {mins:.1f} minutes, {secs} seconds \n"
                msg += f"\tEpoch TrainLoss: {train_loss[0] / batch_num:.4f}  \n"
                msg += f"\tEpoch ValidLoss: {val_loss:.4f}  "
                print(msg)

            if flag:
                break

    def eval(
        self, model: nn.Module, 
        dataloader: DataLoader,
        is_distributed: bool,
        rank: int,
    ) -> torch.FloatTensor:
        model.eval()
        if is_distributed:
            val_loss = torch.zeros(1).to(rank)
        else:
            val_loss = torch.zeros(1).to(self.device)
        if rank == 0:
            inner_pbar = tqdm.tqdm(
                range(len(dataloader)), colour="green", desc="Validation Epoch"
            )
        steps = len(dataloader)
        for step, batch in enumerate(dataloader, start=1):
            if is_distributed:
                output = self._fsdp_step(model, batch, rank, False)
            else:
                output = self._step(model, batch, False)
            loss_b = output.loss
            val_loss[0] += loss_b
            if rank == 0:
                inner_pbar.update(1)
        if is_distributed:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        if rank == 0:
            inner_pbar.close()
        return val_loss[0] / steps

    def _step(self, model: nn.Module, batch: Dict, is_train: bool = True):
        if self.device is not None:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        # mix precision
        with torch.cuda.amp.autocast(dtype=dtype):
            if is_train:
                output = model(**batch)
            else:
                with torch.no_grad():
                    output = model(**batch)
            return output
    
    def _fsdp_step(
        self, model: nn.Module, batch: Dict, rank: int, is_train: bool = True
    ):
        batch = {k: v.to(rank) for k, v in batch.items()}
        if is_train:
            output = model(**batch)
        else:
            with torch.no_grad():
                output = model(**batch)
        return output
    
    def _save(
        self,
        model, 
        total_step: int,
        is_distributed: bool, 
        rank: int
    ) -> None:
        if rank == 0:
            print(f"\n--> entering save model state")
        
        if is_distributed:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = get_lora_state_dict(model)
        else:
            cpu_state = get_lora_state_dict(model)
        
        print(f"saving process: rank {rank}  done w state_dict")
        if rank == 0:
            out_file_name = f"{self.save_file_prefix}-{total_step}.pt"
            out_file_path = os.path.join(self.ckpt_path, out_file_name)
            print(f"--> saving as model name {out_file_name}")
            torch.save(cpu_state, out_file_path)
            files = sorted(
                Path(self.ckpt_path).glob("*.pt"),
                key=lambda x: -int(x.stem.split("-")[-1])
            )
            for file in files[self.max_to_keep:]:
                os.remove(file)