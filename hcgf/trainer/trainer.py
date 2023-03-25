import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pnlp

from ..utils import get_lora_state_dict


class Trainer:

    def __init__(
        self,
        lr: float,
        num_epochs: int,
        warmup_steps: int,
        accumulate_steps: int,
        out_dir: str,
        device: str,
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.accumulate_steps = accumulate_steps
        self.out_dir = out_dir
        self.device = device

    def train(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader
    ) -> None:
        ckpt_path = os.path.join(self.out_dir, "ckpt")
        save_path = os.path.join(self.out_dir, "save")
        pnlp.check_dir(self.out_dir, ckpt_path, save_path)
        print_every = 10
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
        valid_steps = int(batch_num / 3)

        msg = f"Total data batches: {batch_num}, "
        msg += f"Epoch update steps: {epoch_update_steps}, "
        msg += f"Total update steps: {total_update_steps}, "
        msg += f"Warmup update steps: {warmup_update_steps}, "
        msg += f"Validation steps: {valid_steps}, "
        msg += f"Early stop steps: {early_stop_steps}"
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
            train_loss = 0
            print(f"\n\nEpoch {epoch}/{self.num_epochs}")
            for step, batch in enumerate(train_dataloader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # mix precision
                with torch.cuda.amp.autocast():
                    output = model(**batch)
                loss_b = output.loss.detach().float()
                train_loss += loss_b
                loss = output.loss / accumulate_steps
                loss.backward()
                total_step += 1

                if step % accumulate_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if step % print_every == 0:
                    msg = f"Step: {step}, "
                    msg += f"Loss: {train_loss/step:.4f}, "
                    msg += f"LearningRate: {lr_scheduler.get_last_lr()[0]:.6f} "
                    print(msg)

                if total_step % valid_steps == 0 and valid_steps < batch_num:
                    model.eval()
                    val_loss = self.eval(model, dev_dataloader)
                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                        last_improve = total_step
                        out_file_name = f"lora-ckpt-{total_step}.pt"
                        out_file_path = os.path.join(ckpt_path, out_file_name)
                        torch.save(get_lora_state_dict(model), out_file_path)

                    msg = "- " * 30 + "\n"
                    msg += f"Step(Batch)/Epoch: {step}/{epoch},  Total Step: {total_step},  "
                    msg += f"LearningRate: {lr_scheduler.get_last_lr()[0]:.6f}, \n"
                    msg += f"TrainLoss: {train_loss/step:.4f},  "
                    msg += f"ValidLoss: {val_loss:.4f}\n"
                    msg += "= " * 30
                    print(msg)

                    model.train()

                if total_step - last_improve > early_stop_steps:
                    print("Early stop for no improvements...")
                    flag = True

            secs = int(time.perf_counter() - start_time)
            mins = secs / 60
            secs = secs % 60
            model.eval()
            val_loss = self.eval(model, dev_dataloader)
            msg = f"Epoch: {epoch} | time in {mins:.1f} minutes, {secs} seconds \n"
            msg += f"\tEpoch TrainLoss: {train_loss/batch_num:.4f}  \n"
            msg += f"\tEpoch ValidLoss: {val_loss:.4f}  "
            print(msg)
            model.train()

            if flag:
                break

        out_file_name = f"lora-ckpt-{total_step}.pt"
        out_file_path = os.path.join(ckpt_path, out_file_name)
        torch.save(get_lora_state_dict(model), out_file_path)

    def eval(self, model: nn.Module, dataloader: DataLoader) -> float:
        total_loss = 0
        steps = 0
        for step, batch in enumerate(dataloader, start=1):
            steps += 1
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = model(**batch)
            loss_b = output.loss
            total_loss += loss_b
        return total_loss / steps
