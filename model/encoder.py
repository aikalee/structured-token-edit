import torch
import torch.nn as nn

import math
from typing import List, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FeatureEmbedding(nn.Module):
    """
    Combine 3 features into one embedding:
      - token
      - left bracket list
      - right bracket list

    left/right are pooled with mean.
    Final embedding = token_vec + left_vec + right_vec
    """

    def __init__(
        self,
        num_langs: int,
        vocab_size: int,
        num_total_labels: int,
        num_left_labels: int,
        num_right_labels: int,
        d_model: int,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.lang_emb = nn.Embedding(num_langs, d_model)

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        self.base_left_emb = nn.Embedding(num_left_labels, d_model)
        self.base_right_emb = nn.Embedding(num_right_labels, d_model)

        self.overlap_left_emb = nn.Embedding(num_total_labels, d_model)
        self.overlap_right_emb = nn.Embedding(num_total_labels, d_model)
        

        self.d_model = d_model
        self.pad_token_id = pad_token_id

    def _pool_mean(
        self,
        emb_table: nn.Embedding,
        ids_nested: List[List[List[int]]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        ids_nested: B x T x variable_length
        return: [B, T, D]
        """
        batch_vecs = []

        for sent_ids in ids_nested:
            token_vecs = []
            for ids in sent_ids:
                if len(ids) == 0:
                    token_vecs.append(torch.zeros(self.d_model, device=device))
                else:
                    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
                    vecs = emb_table(ids_tensor)      # [k, D]
                    token_vecs.append(vecs.mean(dim=0)) # mean between tokens
            batch_vecs.append(torch.stack(token_vecs, dim=0))  # [T, D]

        return torch.stack(batch_vecs, dim=0)  # [B, T, D]

    def forward(
        self,
        lang_ids: torch.Tensor,
        input_ids: torch.Tensor,
        left_ids: List[List[List[int]]],
        right_ids: List[List[List[int]]],
        base=True,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        left_ids: B x T x variable_length
        right_ids: B x T x variable_length
        """
        device = input_ids.device

        lang_vec = self.lang_emb(lang_ids).unsqueeze(1)
        token_vec = self.token_emb(input_ids)  # [B, T, D]
        
        if base:
            left_vec = self._pool_mean(self.base_left_emb, left_ids, device)    # [B, T, D]
            right_vec = self._pool_mean(self.base_right_emb, right_ids, device) # [B, T, D]
        else:
            left_vec = self._pool_mean(self.overlap_left_emb, left_ids, device)
            right_vec = self._pool_mean(self.overlap_right_emb, right_ids, device)
        

        return lang_vec + token_vec + left_vec + right_vec # sum between features
