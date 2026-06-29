import torch
from tqdm import tqdm
from typing import Optional

from model.base_trainer import BaseTrainer
from model.generation import generate_local_sequence
from model.loss import weighted_structured_token_ce_loss, masked_structured_token_ce_loss, local_bracket_constraint_loss

class DecoderTrainer(BaseTrainer):
    
    def __init__(
        self,
        left_class_weight: Optional[torch.Tensor] = None,
        right_class_weight: Optional[torch.Tensor] = None,
        left_dec_loss_weight: float = 1.0,
        right_dec_loss_weight: float = 1.0,
        structure_loss_weight: float = 0.0,   
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if left_class_weight is not None:
            left_class_weight = left_class_weight.to(self.device)
        if right_class_weight is not None:
            right_class_weight = right_class_weight.to(self.device)
            
        self.left_class_weight = left_class_weight
        self.right_class_weight = right_class_weight

        self.left_dec_loss_weight = left_dec_loss_weight
        self.right_dec_loss_weight = right_dec_loss_weight
        self.structure_loss_weight = structure_loss_weight

        self.config["left_dec_loss_weight"] = left_dec_loss_weight
        self.config["right_dec_loss_weight"] = right_dec_loss_weight
        self.config["structure_loss_weight"] = structure_loss_weight
        

        self.history = {
            "train_loss": [],
            "train_left_dec_loss": [],
            "train_right_dec_loss": [],
            "train_structure_loss": [],
            "dev_loss": [],
            "dev_left_dec_loss": [],
            "dev_right_dec_loss": [],
            "dev_structure_loss": [],
        }
    
    def build_left_right_pairs(self, left2id, right2id):
        pairs = {}
    
        for left_label, left_idx in left2id.items():
            rel = left_label[1:]
            right_label = ")" + rel
    
            if right_label in right2id:
                pairs[left_idx] = right2id[right_label]
        return pairs

    def compute_losses(self, outputs, batch):
        attention_mask = batch["attention_mask"].to(self.device)
        
        # === for decoder ===
        left_dec_logits = outputs["left_dec_logits"]
        right_dec_logits = outputs["right_dec_logits"]
        
        left_dec_target_ids = batch["left_dec_target_ids"].to(self.device)
        right_dec_target_ids = batch["right_dec_target_ids"].to(self.device)

        gate_targets = batch["gate_targets"].to(self.device)

        lr_pair_map = self.build_left_right_pairs(self.left2id, self.right2id)

        special_ids = {self.left2id["<PAD>"], self.left2id["<BOS>"], self.left2id["<EOS>"]}

        if attention_mask.any():
            # left_dec_loss = weighted_structured_token_ce_loss(
            #     left_dec_logits,
            #     left_dec_target_ids,
            #     gate_targets,
            #     pad_id=self.token2id["<PAD>"],
            #     class_weight=self.left_class_weight,
            #     edit_token_weight=1.0,
            #     keep_token_weight=0.0,
            # )
            
            # right_dec_loss = weighted_structured_token_ce_loss(
            #     right_dec_logits,
            #     right_dec_target_ids,
            #     gate_targets,
            #     pad_id=self.token2id["<PAD>"],
            #     class_weight=self.left_class_weight,
            #     edit_token_weight=1.0,
            #     keep_token_weight=0.0,
            # )

            left_dec_loss = masked_structured_token_ce_loss(
                left_dec_logits,
                left_dec_target_ids,
                gate_targets,
                pad_id=self.token2id["<PAD>"],
                class_weight=self.left_class_weight,
                mask_prob=1.0,
            )
            
            right_dec_loss = masked_structured_token_ce_loss(
                right_dec_logits,
                right_dec_target_ids,
                gate_targets,
                pad_id=self.token2id["<PAD>"],
                class_weight=self.left_class_weight,
                mask_prob=1.0,
            )
                
            # structure_loss = local_bracket_constraint_loss(
            #     left_dec_logits,
            #     right_dec_logits,
            #     left_dec_target_ids,
            #     right_dec_target_ids,
            #     attention_mask,
            #     lr_pair_map,
            #     special_ids,
            #     weight_prefix=1.0, 
            #     weight_balance=1.0,
                
            # )

            total_loss = (
                self.left_dec_loss_weight * left_dec_loss +
                self.right_dec_loss_weight * right_dec_loss 
                # self.structure_loss_weight * structure_loss
            )
        else:
            left_dec_loss = left_dec_logits.new_tensor(0.0)         # scalar
            right_dec_loss = right_dec_logits.new_tensor(0.0)
            structure_loss = left_dec_logits.new_tensor(0.0)

        return {
            "loss": total_loss,
            "left_dec_loss": left_dec_loss.detach(),
            "right_dec_loss": right_dec_loss.detach(),
            # "structure_loss": structure_loss.detach(),
        }
    
    def train_one_epoch(self, epoch):
        train_loader = self.build_epoch_loader(epoch)
        self.model.train()

        total_loss = 0.0
        total_left_dec_loss = 0.0
        total_right_dec_loss = 0.0
        total_structure_loss = 0.0


        for batch in tqdm(train_loader, desc="Training"):
            lang_ids = batch["lang_ids"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # left_ids = batch["left_ids"]
            # right_ids = batch["right_ids"]
            base_left_ids = batch["base_left_ids"]
            base_right_ids = batch["base_right_ids"]

            overlap_left_ids = batch["overlap_left_ids"]
            overlap_right_ids = batch["overlap_right_ids"]

            left_dec_input_ids = batch["left_dec_input_ids"].to(self.device)
            right_dec_input_ids = batch["right_dec_input_ids"].to(self.device)
    
            self.optimizer.zero_grad()
    
            outputs = self.model(
                lang_ids=lang_ids,
                input_ids=input_ids,
                # left_ids=left_ids,
                # right_ids=right_ids,
                base_left_ids=base_left_ids,
                base_right_ids=base_right_ids,
                overlap_left_ids=overlap_left_ids,
                overlap_right_ids=overlap_right_ids,
                left_dec_input_ids=left_dec_input_ids,
                right_dec_input_ids=right_dec_input_ids,
                attention_mask=attention_mask,
            )
            losses = self.compute_losses(outputs, batch)
            loss = losses["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                print("bad batch, skip")
                continue
                
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_left_dec_loss += losses["left_dec_loss"].item()
            total_right_dec_loss += losses["right_dec_loss"].item()
            # total_structure_loss += losses["structure_loss"].item()

        n = len(train_loader)
        avg_loss = total_loss / n
        avg_left_dec_loss = total_left_dec_loss / n
        avg_right_dec_loss = total_right_dec_loss / n
        # avg_structure_loss = total_structure_loss / n

        self.history["train_loss"].append(avg_loss)
        self.history["train_left_dec_loss"].append(avg_left_dec_loss)
        self.history["train_right_dec_loss"].append(avg_right_dec_loss)
        # self.history["train_structure_loss"].append(avg_structure_loss)

        return {
            "loss": avg_loss,
            "left_dec_loss": avg_left_dec_loss,
            "right_dec_loss": avg_right_dec_loss,
            # "structure_loss": avg_structure_loss,
        }

    @torch.no_grad()
    def autoregressive_accuracy(self, pred_ids, gold_ids, gate_targets):
        pred_ids = pred_ids[..., 1:]
        S = min(pred_ids.size(-1), gold_ids.size(-1))
        pred_ids = pred_ids[..., :S]
        gold_ids = gold_ids[..., :S]
        # print("pred:", pred_ids, "gold:", gold_ids)

        valid_mask = gold_ids.ne(self.left2id["<PAD>"]) & gold_ids.ne(self.left2id["<BOS>"])
        edit_mask = gate_targets.bool().unsqueeze(-1)

        mask = valid_mask & edit_mask

        correct = (pred_ids == gold_ids) & mask
        return correct.sum().item() / mask.sum().clamp_min(1).item()

    @torch.no_grad()
    def non_empty_rate(self, pred_ids, gate_targets):
        content = (
            pred_ids.ne(self.left2id["<PAD>"]) 
            & pred_ids.ne(self.left2id["<BOS>"]) 
            & pred_ids.ne(self.left2id["<EOS>"])
        )

        non_empty = content.any(dim=-1)
        edit_mask = gate_targets.bool()

        non_empty = non_empty & edit_mask
    
        return non_empty.sum().item() / edit_mask.sum().clamp_min(1).item()
        
    @torch.no_grad()
    def evaluate(self, dataloader=None):
        dataloader = dataloader or self.dev_loader
        if dataloader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_left_dec_loss = 0.0
        total_right_dec_loss = 0.0
        total_structure_loss = 0.0

        total_left_ar_acc = 0.0
        total_right_ar_acc = 0.0

        total_left_non_empty_rate = 0.0
        total_right_non_empty_rate = 0.0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            lang_ids = batch["lang_ids"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # left_ids = batch["left_ids"]
            # right_ids = batch["right_ids"]

            base_left_ids = batch["base_left_ids"]
            base_right_ids = batch["base_right_ids"]

            overlap_left_ids = batch["overlap_left_ids"]
            overlap_right_ids = batch["overlap_right_ids"]

            gate_targets = batch["gate_targets"].to(self.device)
            
            left_dec_input_ids = batch["left_dec_input_ids"].to(self.device)
            right_dec_input_ids = batch["right_dec_input_ids"].to(self.device)

            left_dec_target_ids = batch["left_dec_target_ids"].to(self.device)
            right_dec_target_ids = batch["right_dec_target_ids"].to(self.device)

            outputs = self.model(
                lang_ids=lang_ids,
                input_ids=input_ids,
                # left_ids=left_ids,
                # right_ids=right_ids,
                base_left_ids=base_left_ids,
                base_right_ids=base_right_ids,
                overlap_left_ids=overlap_left_ids,
                overlap_right_ids=overlap_right_ids,
                left_dec_input_ids=left_dec_input_ids,
                right_dec_input_ids=right_dec_input_ids,
                attention_mask=attention_mask,
            )

            losses = self.compute_losses(outputs, batch)

            total_loss += losses["loss"].item()
            total_left_dec_loss += losses["left_dec_loss"].item()
            total_right_dec_loss += losses["right_dec_loss"].item()
            # total_structure_loss += losses["structure_loss"].item()   

            # === generation ===
            left_pred_ids = generate_local_sequence(
                model=self.model,
                batch=batch,
                pad_id=self.left2id["<PAD>"],
                bos_id=self.left2id["<BOS>"],
                eos_id=self.left2id["<EOS>"],
                device=self.device,
                side="left",
                max_len=10,
                edit_mask=gate_targets,
            )

            right_pred_ids = generate_local_sequence(
                model=self.model,
                batch=batch,
                pad_id=self.right2id["<PAD>"],
                bos_id=self.right2id["<BOS>"],
                eos_id=self.right2id["<EOS>"],
                device=self.device,
                side="right",
                max_len=10,
                edit_mask=gate_targets,
            )

            left_ar_acc = self.autoregressive_accuracy(left_pred_ids, left_dec_target_ids, gate_targets)
            right_ar_acc = self.autoregressive_accuracy(right_pred_ids, right_dec_target_ids, gate_targets)
            
            left_non_empty_rate = self.non_empty_rate(left_pred_ids, gate_targets)
            right_non_empty_rate = self.non_empty_rate(right_pred_ids, gate_targets)

            total_left_ar_acc += left_ar_acc
            total_right_ar_acc += right_ar_acc

            total_left_non_empty_rate += left_non_empty_rate
            total_right_non_empty_rate += right_non_empty_rate

        n = len(dataloader)
        avg_loss = total_loss / n
        avg_left_dec_loss = total_left_dec_loss / n
        avg_right_dec_loss = total_right_dec_loss / n
        # avg_structure_loss = total_structure_loss / n
        avg_left_ar_acc = total_left_ar_acc / n
        avg_right_ar_acc = total_right_ar_acc / n
        avg_left_non_empty_rate = total_left_non_empty_rate / n
        avg_right_non_empty_rate = total_right_non_empty_rate / n

        self.history["dev_loss"].append(avg_loss)
        self.history["dev_left_dec_loss"].append(avg_left_dec_loss)
        self.history["dev_right_dec_loss"].append(avg_right_dec_loss)
        # self.history["dev_structure_loss"].append(avg_structure_loss)

        return {
            "loss": avg_loss,
            "left_dec_loss": avg_left_dec_loss,
            "right_dec_loss": avg_right_dec_loss,
            # "structure_loss": avg_structure_loss,
            "left_ar_acc": avg_left_ar_acc,
            "right_ar_acc": avg_right_ar_acc,
            "left_non_empty_rate": avg_left_non_empty_rate,
            "right_non_empty_rate": avg_right_non_empty_rate,
        }

    def fit(self, num_epochs: int, lr_milestones: list, save_dir: Optional[str] = None, save_best_only: bool = True):
        # best_dev_loss = float("inf")
        best_mean_ar_acc = 0.0

        self.print_config()
        if save_dir is not None:
            self.save_config(save_dir, gate=False)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=lr_milestones,
            # milestones=[30, 50, 70, 85],
            gamma=0.5,
        )

        for epoch in range(1, num_epochs + 1):
            print(f"Starting Epoch {epoch}")

            train_stats = self.train_one_epoch(epoch)
            dev_stats = self.evaluate()

            scheduler.step()
            print("current_lr=", self.optimizer.param_groups[0]["lr"])

            if dev_stats is None:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_stats['loss']:.4f} "
                    f"(left={train_stats['left_dec_loss']:.4f}, right={train_stats['right_dec_loss']:.4f}) "
                    # f"structure={train_stats["structure_loss"]:.4f})"
                )
                if save_dir is not None and not save_best_only:
                    self.save_checkpoint(save_dir, f"epoch_{epoch}.pt", epoch, train_stats, None)
            else:
                print(
                    f"Epoch {epoch}: "
                    f"left_ar_acc={dev_stats["left_ar_acc"]:.4f}, "
                    f"right_ar_acc={dev_stats["right_ar_acc"]:.4f}\n" 
                    f"left_non_empty_rate={dev_stats["left_non_empty_rate"]:.4f}, "
                    f"right_non_empty_rate={dev_stats["right_non_empty_rate"]:.4f}\n" 
                    f"train_loss={train_stats['loss']:.4f} "
                    f"(left={train_stats['left_dec_loss']:.4f}, right={train_stats['right_dec_loss']:.4f}) "
                    # f"structure={train_stats["structure_loss"]:.4f})\n" 
                    f"dev_loss={dev_stats['loss']:.4f} "
                    f"(left={dev_stats['left_dec_loss']:.4f}, right={dev_stats['right_dec_loss']:.4f}) "
                    # f"structure={dev_stats["structure_loss"]:.4f})" 
                )

                if save_dir is not None:
                    if save_best_only:
                        # if dev_stats["loss"] < best_dev_loss:
                        mean_ar_acc = (dev_stats["left_ar_acc"] + dev_stats["right_ar_acc"]) / 2
                        if mean_ar_acc > best_mean_ar_acc:
                            # best_dev_loss = dev_stats["loss"]
                            best_mean_ar_acc = mean_ar_acc
                            self.save_checkpoint(
                                save_dir, 
                                f"{self.lang}_decoder.pt", 
                                epoch, 
                                train_stats, 
                                dev_stats
                            )
                    else:
                        self.save_checkpoint(
                            save_dir, 
                            f"{self.lang}_decoder@epoch_{epoch}.pt", 
                            epoch, 
                            train_stats, 
                            dev_stats
                        )

        
