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



class FishBoutEncoder(nn.Module):
    """
    BERT-style masked model for continuous targets.

    - Uses only an encoder (bidirectional) like BERT.
    - During training you should provide `mask_positions` indicating which
      time steps should be predicted. The model replaces those positions
      with a learned `mask_embedding` and predicts the original values.

    Forward:
      src: (batch, seq_len, input_dim)
      mask_positions: optional bool tensor (batch, seq_len) where True means
                      that position is masked and should be predicted.

    Returns:
      preds: (batch, seq_len, target_dim) -- model predictions for every position.
             Typically you compute loss only at masked positions.
    """

    def __init__(self, input_dim, seq_len, d_model, nhead,
                 num_layers, dim_feedforward, dropout):
        super().__init__()
        
        # Input projection to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)

        # Learnable mask embedding (used to replace masked positions)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

        # Transformer encoder (bidirectional attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, 
                                                   batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, input_dim)

    def forward(self, src, mask_positions=None):
        """
        src: (batch, seq_len, input_dim)
        mask_positions: None or (batch, seq_len) bool tensor
        """
        x = self.input_proj(src)  # -> (batch, seq_len, d_model)
        # print(f"x after projection")
        # print(f"{x.shape=}")
        # print(f"{x=}")
        x = self.pos_encoder(x)

        if mask_positions is not None:
            # mask_positions: bool (batch, seq_len)
            # replace masked position embeddings with mask_embedding
            # Expand mask_embedding to (batch, seq_len, d_model)
            mask_tok = self.mask_embedding.view(1, 1, -1)
            mask_tok = mask_tok.expand(x.size(0), x.size(1), -1)

            # Create a float mask for blending
            m = mask_positions.unsqueeze(-1).to(x.dtype)  # (batch, seq_len, 1)
            x = x * (1.0 - m) + mask_tok * m

        memory = self.encoder(x)  # (batch, seq_len, d_model)
        preds = self.output(memory)  # (batch, seq_len, target_dim)
        return preds