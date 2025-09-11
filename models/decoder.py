import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    GPT-style causal model for next-bout prediction.

    - Uses Transformer decoder blocks (causal masking).
    - Predicts the next bout embedding at each timestep.
    - At inference: can autoregressively generate sequences.

    Forward:
      src: (batch, seq_len, input_dim) - observed bouts

    Returns:
      preds: (batch, seq_len, target_dim) - predicted next bout for each position.
              preds[:, t] is the model's prediction for src[:, t+1]
    """

    def __init__(self, input_dim, target_dim, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1, seq_len=5):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)

        # Transformer decoder (causal self-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output = nn.Linear(d_model, target_dim)

    def _generate_square_subsequent_mask(self, sz: int, device):
        """Generates an upper-triangular matrix of -inf for masking future positions."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src):
        """
        src: (batch, seq_len, input_dim)
        """
        x = self.input_proj(src)  # -> (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # Causal mask to prevent seeing future tokens
        tgt_mask = self._generate_square_subsequent_mask(x.size(1), x.device)

        # TransformerDecoder in PyTorch expects:
        #   tgt: target sequence (queries)
        #   memory: encoder output (keys/values)
        # For GPT-like use, we set memory = None and use tgt as both
        decoded = self.decoder(tgt=x, memory=None, tgt_mask=tgt_mask)

        preds = self.output(decoded)  # (batch, seq_len, target_dim)
        return preds
