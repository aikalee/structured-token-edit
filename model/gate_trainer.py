import torch
from tqdm import tqdm
from typing import Optional

from model.base_trainer import BaseTrainer
from model.loss import weighted_gate_bce_loss, masked_gate_bce_loss, focal_gate_bce_loss


class GateTrainer(BaseTrainer):
    def __init__(self, gate_threshold, **kwargs):
        
        super().__init__(**kwargs)
        
        self.history = {
            "train_loss": [],
            "dev_loss": [],
        }
        self.gate_threshold = gate_threshold
        self.config["gate_threshold"] = gate_threshold
        
    def compute_losses(self, outputs, batch):
        attention_mask = batch["attention_mask"].to(self.device)
        
        # === for correction gate ===
        gate_logits = outputs["gate_logits"]
        gate_targets = batch["gate_targets"].to(self.device)

        if attention_mask.any():
            # loss = weighted_gate_bce_loss(
            #     gate_logits,
            #     gate_targets,
            #     attention_mask,
            # )
            
            # loss = focal_gate_bce_loss(
            #     gate_logits, 
            #     gate_targets, 
            #     attention_mask, 
            #     alpha=0.75, 
            #     gamma=2.0
            # )

            loss = masked_gate_bce_loss(
                gate_logits,
                gate_targets,
                attention_mask,
                no_edit_keep_prob=0.3,
                pos_weight=5.0,
                fp_lambda=0.1,
            )

        else:
             loss = logits.new_tensor(0.0)

        return loss

    def train_one_epoch(self, epoch):
        train_loader = self.build_epoch_loader(epoch)
        self.model.train()

        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            lang_ids = batch["lang_ids"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            base_left_ids = batch["base_left_ids"]
            base_right_ids = batch["base_right_ids"]

            overlap_left_ids = batch["overlap_left_ids"]
            overlap_right_ids = batch["overlap_right_ids"]

            # print(base_left_ids[0])
            # print(overlap_left_ids[0])
            
            self.optimizer.zero_grad()
    
            outputs = self.model(
                lang_ids=lang_ids,
                input_ids=input_ids,
                base_left_ids=base_left_ids,
                base_right_ids=base_right_ids,
                overlap_left_ids=overlap_left_ids,
                overlap_right_ids=overlap_right_ids,
                attention_mask=attention_mask,
            )
            loss = self.compute_losses(outputs, batch)
            total_loss += loss.item()
            
            loss.backward()
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        n = len(train_loader)
        avg_loss = total_loss / n
        self.history["train_loss"].append(avg_loss)

        return avg_loss
        
    @torch.no_grad()
    def non_zero_rate(self, gate_prob, attention_mask):
        gate_prob = gate_prob * attention_mask
        pred_gate = gate_prob > self.gate_threshold
        return (pred_gate == 1).sum().item() / (gate_prob != 0).sum().item()

    @torch.no_grad()
    def gate_f1(self, gate_prob, gate_targets, gate_threshold, attention_mask):
        mask = attention_mask.bool()
        
        pred_edit = (gate_prob > gate_threshold) & mask
        pred_keep = (~pred_edit) & mask
        
        gold_edit = (gate_targets == 1) & mask
        gold_keep = (gate_targets == 0) & mask
        
    
        tp = (pred_edit & gold_edit).sum().item()
        fp = (pred_edit & gold_keep).sum().item()
        fn = (pred_keep & gold_edit).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
        return precision, recall, f1
        
    @torch.no_grad()
    def evaluate(self, dataloader=None):
        dataloader = dataloader or self.dev_loader
        if dataloader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_non_zero_rate = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0


        for batch in tqdm(dataloader, desc="Evaluating"):
            lang_ids = batch["lang_ids"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            gate_targets = batch["gate_targets"].to(self.device)

            base_left_ids = batch["base_left_ids"]
            base_right_ids = batch["base_right_ids"]

            overlap_left_ids = batch["overlap_left_ids"]
            overlap_right_ids = batch["overlap_right_ids"]

            outputs = self.model(
                lang_ids=lang_ids,
                input_ids=input_ids,
                base_left_ids=base_left_ids,
                base_right_ids=base_right_ids,
                overlap_left_ids=overlap_left_ids,
                overlap_right_ids=overlap_right_ids,
                attention_mask=attention_mask,
            )

            # print(outputs["gate_logits"].min().item(),
            #       outputs["gate_logits"].max().item(),
            #       outputs["gate_logits"].mean().item())
            
            # print(outputs["gate_prob"].min().item(),
            #       outputs["gate_prob"].max().item(),
            #       outputs["gate_prob"].mean().item())

            loss = self.compute_losses(outputs, batch)
            total_loss += loss.item()
            
            non_zero_rate = self.non_zero_rate(outputs["gate_prob"],  attention_mask)
            total_non_zero_rate += non_zero_rate

            
            precision, recall, f1 = self.gate_f1(outputs["gate_prob"], gate_targets, self.gate_threshold, attention_mask)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            

        n = len(dataloader)
        avg_loss = total_loss / n
        avg_non_zero_rate = total_non_zero_rate / n
        avg_precision = total_precision / n
        avg_recall = total_recall / n
        avg_f1 = total_f1 / n
        self.history["dev_loss"].append(avg_loss)
        
        return {
            "loss": avg_loss,
            "non_zero_rate": avg_non_zero_rate,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }
        
    def fit(self, num_epochs: int, save_dir: Optional[str] = None, save_best_only: bool = True):
        self.print_config()
        # best_dev_loss = float("inf")
        best_precision = 0.0

        if save_dir is not None:
            self.save_config(save_dir, gate=True)


        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[30, 50, 70, 85],
            gamma=0.5,
        )

        for epoch in range(1, num_epochs + 1):
            print(f"Starting Epoch {epoch}")

            train_loss = self.train_one_epoch(epoch)
            dev_stats = self.evaluate()

            scheduler.step()
            print("current_lr=", self.optimizer.param_groups[0]["lr"])

            if dev_stats is None:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.4f} "
                )
                if save_dir is not None and not save_best_only:
                    self.save_checkpoint(save_dir, f"epoch_{epoch}.pt", epoch, train_loss, None)
            else:
                print(
                    f"Epoch {epoch}: " 
                    f"train_loss={train_loss:.4f}, "
                    f"dev_loss={dev_stats["loss"]:.4f}\n"
                    f"non_zero_rate={dev_stats["non_zero_rate"]:.4f}\n"
                    f"precision={dev_stats["precision"]:.4f}, "
                    f"recall={dev_stats["recall"]:.4f}, "
                    f"f1={dev_stats["f1"]:.4f} "
                )

                if save_dir is not None:
                    if save_best_only:
                        if dev_stats["precision"] > best_precision:
                        # if dev_stats["loss"] < best_dev_loss:
                            # best_dev_loss = dev_stats["loss"]
                            best_precision = dev_stats["precision"]
                            self.save_checkpoint(
                                save_dir, 
                                f"gate.pt", 
                                epoch, 
                                {"loss": train_loss}, 
                                dev_stats
                            )
                    else:
                        self.save_checkpoint(
                            save_dir, 
                            f"gate@epoch_{epoch}.pt", 
                            epoch, 
                            {"loss": train_loss}, 
                            dev_stats
                        )

            

        
