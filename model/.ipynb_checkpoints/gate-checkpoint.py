import torch
import torch.nn as nn

import math
from typing import List, Optional
from model.encoder import PositionalEncoding, FeatureEmbedding

class StructuredTokenGate(nn.Module):
    def __init__(
        self,
        num_langs: int,
        vocab_size: int,
        num_total_labels: int,
        num_left_labels: int,
        num_right_labels: int,
        device: torch.device,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        pad_bracket_id: int = 0,
        bos_bracket_id: int = 1,
        eos_bracket_id: int = 2,
        max_len: int = 2048,
     ):
        super().__init__()
        self.config = {
            "num_langs": num_langs,
            "vocab_size": vocab_size,
            "num_total_labels": num_total_labels,
            "num_left_labels": num_left_labels,
            "num_right_labels": num_right_labels,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "pad_token_id": pad_token_id,
            "pad_bracket_id": pad_bracket_id,
            "bos_bracket_id": bos_bracket_id,
            "eos_bracket_id": eos_bracket_id,
            "max_len": max_len,
        }

        # === encoder ===
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        # self.num_total_labels = num_total_labels
        # self.num_left_labels = num_left_labels
        # self.num_right_labels = num_right_labels
    
        self.feature_emb = FeatureEmbedding(
            num_langs=num_langs,
            vocab_size=vocab_size,
            num_total_labels=num_total_labels,
            num_left_labels=num_left_labels,
            num_right_labels=num_right_labels,
            d_model=d_model,
            pad_token_id=pad_token_id,
        )
    
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)
    
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.src_fuse = nn.Linear(self.d_model * 2, self.d_model)

        # === correction gate ===
        # self.correction_gate = nn.Linear(d_model, 1)
        self.correction_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
         
    def forward(
        self,
        lang_ids: torch.Tensor,
        input_ids: torch.Tensor,
        base_left_ids: List[List[List[int]]],
        base_right_ids: List[List[List[int]]],
        overlap_left_ids: List[List[List[int]]],
        overlap_right_ids: List[List[List[int]]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:

        outputs = {}
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # token + left + right
        base_x = self.feature_emb(
            lang_ids, 
            input_ids, 
            base_left_ids, 
            base_right_ids, 
            base=True
        )  # [B, T, D]
        base_x = base_x * math.sqrt(self.d_model)
        base_x = self.pos_enc(base_x)
        base_x = self.emb_dropout(base_x)

        if overlap_left_ids is not None and overlap_right_ids is not None:
            overlap_x = self.feature_emb(
                lang_ids, 
                input_ids, 
                overlap_left_ids, 
                overlap_right_ids, 
                base=False
            )  # [B, T, D]
            overlap_x = overlap_x * math.sqrt(self.d_model)
            overlap_x = self.pos_enc(overlap_x)
            overlap_x = self.emb_dropout(overlap_x)
    
            x = torch.cat([base_x, overlap_x], dim=-1)
            x = self.src_fuse(x)
        else:
            x = base_x

        # encoder output 
        hidden_states = self.encoder(
            x,
            src_key_padding_mask=(attention_mask == 0),
        )

        # === correction gate ===
        # gate_inputs = hidden_states.detach()       # gate independent from encoder
        gate_logits = self.correction_gate(hidden_states).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        outputs["gate_logits"] = gate_logits
        outputs["gate_prob"] = gate_prob
        
        return outputs
    