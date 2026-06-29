import torch

@torch.no_grad()
def generate_local_sequence(model, batch,  pad_id, bos_id, eos_id, device, side="left", max_len=10, edit_mask=None):
        lang_ids = batch["lang_ids"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # left_ids = batch["left_ids"]
        # right_ids = batch["right_ids"]
        base_left_ids = batch["base_left_ids"]
        base_right_ids = batch["base_right_ids"]

        overlap_left_ids = batch["overlap_left_ids"]
        overlap_right_ids = batch["overlap_right_ids"]
        
        # device = input_ids.device
        B, T = input_ids.shape

        dec_input_ids = torch.full(
            (B, T, 1),
            bos_id,                   # initial bos
            dtype=torch.long,
            device=device,
        )

        if edit_mask is None:
            edit_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            edit_mask = edit_mask.to(device).bool()
            
        finished = ~edit_mask

        # finished = torch.zeros(B, T, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            if side == "left":
                outputs = model(
                    lang_ids=lang_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # left_ids=left_ids,
                    # right_ids=right_ids,
                    base_left_ids=base_left_ids,
                    base_right_ids=base_right_ids,
                    overlap_left_ids=overlap_left_ids,
                    overlap_right_ids=overlap_right_ids,
                    left_dec_input_ids=dec_input_ids, # Why?
                    right_dec_input_ids=None,
                )
                logits = outputs["left_dec_logits"]
            else:
                outputs = model(
                    lang_ids=lang_ids,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    base_left_ids=base_left_ids,
                    base_right_ids=base_right_ids,
                    overlap_left_ids=overlap_left_ids,
                    overlap_right_ids=overlap_right_ids,
                    left_dec_input_ids=None,
                    right_dec_input_ids=dec_input_ids,
                )
                logits = outputs["right_dec_logits"]
            next_logits = logits[:, :, -1, :]          # last element in S dimension: [B, T, D]
            next_ids = next_logits.argmax(dim=-1)      # largest label in D dimension: [B, T]
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, pad_id),
                next_ids,
            )

            dec_input_ids = torch.cat(
                [dec_input_ids, next_ids.unsqueeze(-1)],
                dim=-1
            )

            finished = finished | (next_ids == eos_id)

            if finished.all():
                break
        return dec_input_ids