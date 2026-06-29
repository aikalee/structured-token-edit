import torch
import torch.nn as nn

import math
from typing import List, Optional
from model.encoder import FeatureEmbedding, PositionalEncoding

class StructuredTokenDecoder(nn.Module):
    """
    Dual-head token-level decoder for per-token sequence prediction
    """
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
        decoder_layers: int = 1,
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
            "decoder_layers": decoder_layers,
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
        
        # === token-level decoder ===
        self.pad_bracket_id = pad_bracket_id
        self.bos_bracket_id = bos_bracket_id
        self.eos_bracket_id = eos_bracket_id

        self.left_vocab_size = num_left_labels 
        self.right_vocab_size = num_right_labels

        self.left_bracket_emb = nn.Embedding(
            self.left_vocab_size,
            d_model,
            padding_idx=self.pad_bracket_id,
        )
        self.right_bracket_emb = nn.Embedding(
            self.right_vocab_size,
            d_model,
            padding_idx=self.pad_bracket_id,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.left_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_layers,
        )
        self.right_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        self.left_decoder_out = nn.Linear(d_model, self.left_vocab_size)
        self.right_decoder_out = nn.Linear(d_model, self.right_vocab_size)

    def _decode_brackets(
        self,
        hidden_states: torch.Tensor,
        dec_input_ids: torch.Tensor,
        emb: nn.Embedding,
        decoder: nn.TransformerEncoder,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        
        B, T, D = hidden_states.shape
        _, _, S = dec_input_ids.shape

        x = emb(dec_input_ids)
        ctx = hidden_states.unsqueeze(2)
        x = x + ctx

        x = x.reshape(B * T, S, D)

        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        pad_mask = dec_input_ids.reshape(B * T, S).eq(self.pad_bracket_id)
        out = decoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )
        logits = out_proj(out)
        logits = logits.reshape(B, T, S, -1)       # [B * T, S, D] -> [B, T, S, D]
        return logits
        
    def forward(
        self,
        lang_ids: torch.Tensor,
        input_ids: torch.Tensor,
        # left_ids: List[List[List[int]]],
        # right_ids: List[List[List[int]]],
        base_left_ids: List[List[List[int]]],
        base_right_ids: List[List[List[int]]],
        overlap_left_ids: List[List[List[int]]],
        overlap_right_ids: List[List[List[int]]],
        left_dec_input_ids: Optional[torch.Tensor] = None,
        right_dec_input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
    
        """
        input: hidden_states [B, T, D]
        output: gate_logits  [B, T]
                gate_prob    [B, T]
        """

        outputs = {}
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # token + left + right
        # x = self.feature_emb(lang_ids, input_ids, left_ids, right_ids)  # [B, T, D]
        # x = x * math.sqrt(self.d_model)
        # x = self.pos_enc(x)
        # x = self.emb_dropout(x)


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

        # === token-level decoder ===
        if left_dec_input_ids is not None:
            left_dec_logits = self._decode_brackets(
                hidden_states=hidden_states,            # encoder output
                dec_input_ids=left_dec_input_ids,
                emb=self.left_bracket_emb,              # decoder input embedding
                decoder=self.left_decoder,
                out_proj=self.left_decoder_out,
            )
            outputs["left_dec_logits"] = left_dec_logits
            
        if right_dec_input_ids is not None:
            right_dec_logits = self._decode_brackets(
                hidden_states=hidden_states,
                dec_input_ids=right_dec_input_ids,
                emb=self.right_bracket_emb,
                decoder=self.right_decoder,
                out_proj=self.right_decoder_out,
            )
            outputs["right_dec_logits"] = right_dec_logits
        
        return outputs