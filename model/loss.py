import torch
import torch.nn.functional as F

def weighted_structured_token_ce_loss(
    logits,
    targets,
    gate_targets,
    pad_id=0,
    class_weight=None,
    edit_token_weight=1.0,
    keep_token_weight=0.3,
):
    B, T, S, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        weight=class_weight,
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, T, S)

    valid = targets.ne(pad_id).float()

    token_weight = torch.where(
        gate_targets.bool(),
        torch.full_like(gate_targets, edit_token_weight),
        torch.full_like(gate_targets, keep_token_weight),
    ).unsqueeze(-1)

    loss = (loss * valid * token_weight).sum() / (
        valid * token_weight
    ).sum().clamp(min=1.0)

    return loss

def masked_structured_token_ce_loss(
    logits,
    targets,
    gate_targets,
    pad_id=0,
    class_weight=None,
    mask_prob=0.3,
):
    B, T, S, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        weight=class_weight,
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, T, S)

    valid_mask = targets.ne(pad_id)
    edit_mask = gate_targets.bool()
    edit_mask = edit_mask.unsqueeze(-1)

    keep_sample = torch.rand(B, T, 1, device=logits.device) < mask_prob

    loss_mask = valid_mask & (edit_mask | keep_sample)
    loss = loss * loss_mask.float()
    denom = loss_mask.float().sum().clamp_min(1.0)

    return loss.sum() / denom


def get_batch_lr_pairs(left_target_ids, right_target_ids, lr_pair_map, special_ids):
    left_present = set(left_target_ids.detach().cpu().reshape(-1).tolist()) - special_ids
    right_present = set(right_target_ids.detach().cpu().reshape(-1).tolist()) - special_ids

    pairs = []

    for open_id, close_id in lr_pair_map.items():
        if open_id in left_present or close_id in right_present:
            pairs.append((open_id, close_id))
    return pairs

def local_bracket_constraint_loss(
    left_dec_logits, 
    right_dec_logits, 
    left_dec_target_ids,
    right_dec_target_ids,
    attention_mask, 
    lr_pair_map,
    special_ids,
    weight_prefix=1.0, 
    weight_balance=1.0,
):
    """
    left_dec_logits: [B, T, S_left, V_left]
    right_dec_logits: [B, T, S_right, V_right]
    attention_mask: [B, T]
    """
    
    batch_pairs = get_batch_lr_pairs(
        left_dec_target_ids,
        right_dec_target_ids,
        lr_pair_map,
        special_ids=special_ids,
    )

    
    left_probs = torch.softmax(left_dec_logits, dim=-1)    # [B, T, S, V]
    right_probs = torch.softmax(right_dec_logits, dim=-1)  # [B, T, S, V]

    mask = attention_mask.float()

    total_prefix_loss = left_dec_logits.new_tensor(0.0)
    total_balance_loss = left_dec_logits.new_tensor(0.0)

    for open_id, close_id in batch_pairs: 
        open_prob = left_probs[..., open_id]               # [B, T, S]
        close_prob = right_probs[..., close_id]

        open_count = open_prob.sum(dim=-1) * mask          # combine open prob in sequences: [B, T, S] -> [B, T] 
        close_count = close_prob.sum(dim=-1) * mask 

        open_cum = open_count.cumsum(dim=1)                # [a, b, c] -> [a, a+b, a+b+c]: [B, T]
        close_cum = close_count.cumsum(dim=1)

        prefix_violation = F.relu(close_cum - open_cum)    # [B, T]
        prefix_loss = (prefix_violation * mask).sum() / mask.sum().clamp(min=1.0)

        total_prefix_loss += prefix_loss

        # === balance loss ===

        balance_loss = abs(open_count - close_count).mean()
        total_balance_loss += balance_loss

    return weight_prefix * total_prefix_loss + weight_balance * total_balance_loss


def weighted_gate_bce_loss(
    gate_logits,
    gate_targets,
    attention_mask,
):
    """
    gate_logits: [B, T]
    gate_targets: [B, T]
    attention_mask: [B, T]
    """
    # avoid class imbalance between 1 and 0
    with torch.no_grad():
        num_edit = (gate_targets * attention_mask).sum()
        num_keep = ((1.0 - gate_targets) * attention_mask).sum()

        # pos_weight = (num_keep / num_edit.clamp(min=1.0)).clamp(max=20.0)
        pos_weight = gate_targets.new_tensor(1.0)

    loss_raw = F.binary_cross_entropy_with_logits(
        gate_logits,
        gate_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    mask = attention_mask.float()
    loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)

    return loss

def masked_gate_bce_loss(
    gate_logits,
    gate_targets,
    attention_mask,
    no_edit_keep_prob=0.3,
    pos_weight=None,
    fp_lambda=0.05,
):
    gate_targets = gate_targets.float()
    valid_mask = attention_mask.bool()

    edit_mask = (gate_targets == 1) & valid_mask
    no_edit_mask = (gate_targets == 0) & valid_mask

    sampled_no_edit = (
        no_edit_mask
        & (torch.rand_like(gate_targets) < no_edit_keep_prob)
    )
    loss_mask = edit_mask | sampled_no_edit
    if pos_weight is None:
        pos_weight = gate_logits.new_tensor(1.0)
    else:
        pos_weight = torch.tensor([pos_weight], device=gate_logits.device)

    loss_raw = F.binary_cross_entropy_with_logits(
        gate_logits,
        gate_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    bce = (loss_raw * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    prob = torch.sigmoid(gate_logits)
    neg = 1.0 - gate_targets

    fp_penalty = (prob * neg * loss_mask).sum() / (neg * loss_mask).sum().clamp(min=1.0)

    return bce + fp_lambda * fp_penalty

    # loss = (loss_raw * loss_mask.float()).sum()
    # loss = loss / loss_mask.float().sum().clamp_min(1.0)

    # return loss

def focal_gate_bce_loss(gate_logits, gate_targets, mask, alpha=0.25, gamma=2.0):
    gate_targets = gate_targets.float()
    bce = F.binary_cross_entropy_with_logits(
        gate_logits,
        gate_targets,
        reduction="none",
    )
    prob = torch.sigmoid(gate_logits)
    pt = torch.where(gate_targets == 1, prob, 1 - prob)

    focal = (1 - pt).pow(gamma)
    alpha_t = torch.where(
        gate_targets == 1,
        torch.full_like(gate_targets, alpha),
        torch.full_like(gate_targets, 1 - alpha),
    )
    loss = alpha_t * focal * bce
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum().clamp_min(1.0)
    