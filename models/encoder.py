import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # broadcasting
        #print(f"{self.pe[:, :x.size(1), :]=}")
        return self.dropout(x)



class TransformerEncoder(nn.Module):
    """
    BERT-style encoder with bidirectional attention.

    Forward:
      src: (batch, seq_len, input_dim)
      mask_positions: (batch, seq_len) where True means
                      that position is masked and should be predicted.

    Returns:
      preds: (batch, seq_len, target_dim) -- model predictions for every position.
             (but the loss will be computed only on masked positions)
    """

    def __init__(self, input_dim, seq_len, d_model, nhead,
                 num_layers, dropout, learnable_mask_embedding):
        super().__init__()
        
        # Input projection to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)

        # Learnable mask embedding (used to replace masked positions)
        self.mask_embedding = None
        if learnable_mask_embedding:
            self.mask_embedding = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

        dim_feedforward = d_model * 4  # commonly used value
        
        # Transformer encoder (bidirectional attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, 
                                                   batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, input_dim)

    def forward(self, src, mask_positions, return_embeddings=False):
        """
        src: (batch, seq_len, input_dim)
        mask_positions: (batch, seq_len) bool tensor
        """
        x = self.input_proj(src)  # -> (batch, seq_len, d_model)
        # print(f"x after projection")
        # print(f"{x.shape=}")
        # print(f"{x=}")
        x = self.pos_encoder(x)

        if self.mask_embedding is not None:
            # mask_positions: bool (batch, seq_len)
            # replace masked position embeddings with mask_embedding
            mask_tok = self.mask_embedding.view(1, 1, -1)
            mask_tok = mask_tok.expand(x.size(0), x.size(1), -1)

            # Create a float mask for blending
            m = mask_positions.unsqueeze(-1).to(x.dtype)  # (batch, seq_len, 1)
            x = x * (1.0 - m) + mask_tok * m

        memory = self.encoder(x)  # (batch, seq_len, d_model)
        preds = self.output(memory)  # (batch, seq_len, target_dim)
        if return_embeddings:
            return preds, memory
        else:
            return preds