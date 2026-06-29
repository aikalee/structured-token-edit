import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

class BaseTrainer:
    def __init__(
        self,
        lang,
        model,
        edit_dataset,
        no_edit_dataset,
        batch_size,
        collate_fn,
        no_edit_schedule,
        no_edit_sample_mode,
        # no_edit_ratio,
        # warmup_epochs,
        dev_loader,
        token2id,
        left2id,
        right2id,
        device: Optional[torch.device] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,   
    ):
        self.lang = lang 
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # self.train_loader = train_loader
        self.edit_dataset = edit_dataset
        self.no_edit_dataset = no_edit_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        self.no_edit_schedule = no_edit_schedule
        self.no_edit_sample_mode = no_edit_sample_mode
        
        # self.warmup_epochs = warmup_epochs
        # self.no_edit_ratio = no_edit_ratio
        
        self.dev_loader = dev_loader

        self.token2id = token2id
        self.left2id = left2id
        self.right2id = right2id

        self.grad_clip = grad_clip
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.config = {
            "lang": lang,
            "device": str(self.device),
            "no_edit_schedule": no_edit_schedule,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
        }
            
    def print_config(self):
        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            print(
                "Model config",
                json.dumps(model_config, indent=2),
                "\nTraining config",
                json.dumps(self.config, indent=2),
            )
        else:
            print("Training config", json.dumps(self.config, indent=2))

    def get_no_edit_ratio_for_epoch(self, epoch):
        for start_epoch, ratio in reversed(self.no_edit_schedule):
            if epoch >= start_epoch:
                return ratio
        return 0.0

    def build_epoch_loader(self, epoch):
        ratio = self.get_no_edit_ratio_for_epoch(epoch)

        if ratio is None:

            dataset = torch.utils.data.ConcatDataset([
                self.edit_dataset,
                self.no_edit_dataset,
            ])
        elif ratio == 0:
            dataset = self.edit_dataset
        else:
            if self.no_edit_sample_mode == "relative_to_edit":
                n_no_edit = int(len(self.edit_dataset) * ratio)
            elif self.no_edit_sample_mode == "fraction_of_no_edit":
                n_no_edit = int(len(self.no_edit_dataset) * ratio)

            if self.no_edit_sample_mode not in ["relative_to_edit", "fraction_of_no_edit"]:
                raise ValueError(
                    "Wrong mode, no_edit_sample_mode must be \'relative_to_edit\' or \'fraction_of_no_edit\'"
                )
         
            # n_no_edit = min(
            #     len(self.no_edit_dataset),
            #     int(len(self.edit_dataset) * ratio),
            # )
            indices = torch.randperm(len(self.no_edit_dataset))[:n_no_edit].tolist()
            no_edit_subset = torch.utils.data.Subset(
                self.no_edit_dataset,
                indices,
            )
            dataset = torch.utils.data.ConcatDataset([
                self.edit_dataset,
                no_edit_subset,
            ])

        # print("ratio:", ratio, type(ratio))
        # print("edit len:", len(self.edit_dataset))
        # print("no_edit len:", len(self.no_edit_dataset))
        # print("n_no_edit:", n_no_edit if ratio not in [None, 0] else None)
        # print("dataset len:", len(dataset))
        # print("batch size:", self.batch_size)
        # print("loader batches:", len(DataLoader(dataset, batch_size=self.batch_size)))
        
        return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
                        
    def save_checkpoint(self, save_dir: str, filename: str, epoch: int, train_stats: dict, dev_stats: Optional[dict]):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_stats": train_stats,
            "dev_stats": dev_stats,
            "history": self.history,
            "left2id": self.left2id,
            "right2id": self.right2id,
        }

        torch.save(ckpt, save_path / filename)
        print(f"Checkpoint saved at {save_path / filename}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "history" in ckpt:
            self.history = ckpt["history"]

        return ckpt

    def save_config(self, save_dir: str, gate=True):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "model": getattr(self.model, "config", {}),
            "training": self.config,
        }

        config_name = f"{self.lang}_gate_config.json" if gate else f"{self.lang}_decoder_config.json"

        with open(Path(save_path) / config_name, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
    