from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Embedded dimension for the tokens (table cells)
    embedded_dimension: int = 64

    # Number of layers in the transformer
    n_layers: int = 12

    # Number of heads used for attention
    n_heads: int = 2

    # Hidden dimension in the feed-forward MLP
    hidden_dimension_ff: int = 256

    # Hidden dimension used by the encoder
    hidden_dimension_enc: int = 256

    # Hidden dimension used by the decoder
    hidden_dimension_dec: int = 256

    # Number of features to be grouped for tokenization
    feature_group_size: int = 3

    # Thinking rows added to the transformer input
    n_thinking_rows: int = 16

    # Number of buckets to discretisize the real number line
    n_buckets: int = 32

    # Temperature adjustment knob
    temperature: float = 1.0

    # Device to be used for inference
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, emb_dim: int):
        super().__init__()

        # Dimensions sanity check
        if emb_dim % n_heads != 0:
            raise RuntimeError("The embedded dimension must be divisible by the number of heads.")
        pass

    def forward(self, attn_in: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        pass


class FeedForward(nn.Module):

    def __init__(self, embedded_dim: int, hidden_dim: int):
        super().__init__()

        # MLP that will perform the feed-forward step
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # Do feed-forward step
        return self.mlp(emb)
    

class Layer(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Define attention on each row between features
        self.feature_attn = MultiHeadSelfAttention(config.n_heads, config.embedded_dimension)
        self.layer_norm_fa = nn.LayerNorm(config.embedded_dimension)

        # Define attention on each feature between rows
        self.row_attn = MultiHeadSelfAttention(config.n_heads, config.embedded_dimension)
        self.layer_norm_ra = nn.LayerNorm(config.embedded_dimension)

        # Define feed forward step
        self.feed_forward = FeedForward(config.embedded_dimension, config.hidden_dimension_ff)
        self.layer_norm_ff = nn.LayerNorm(config.embedded_dimension)

    def forward(self, emb: torch.Tensor, test_size: int) -> torch.Tensor:
        B, S, Fg1, E = emb.shape # (B, S, Fg+1, emb_dim)

        # Reshape to create feature attention input
        f_attn_in = emb.reshape([B*S, Fg1, E]) # (B*S, Fg+1, emb_dim)
        # Run feture attention
        f_attn_out = self.feature_attn(f_attn_in) # (B*S, Fg+1, emb_dim)
        # Add residual and layer normalize
        res_f_attn_out = self.layer_norm_fa(f_attn_out + f_attn_in) # (B*S, Fg+1, emb_dim)

        # Create attention mask for row attention (test attends to train, train attends to train)
        mask = torch.zeros([S, S], dtype=torch.float32, device=emb.device)
        mask[:,-test_size:] = float('-inf')

        # Reshape and transpose to create row attention input
        r_attn_in = res_f_attn_out.reshape([B, S, Fg1, E]).transpose(1,2).reshape([B*Fg1, S, E]) # (B*(Fg+1), S, emb_dim)
        # Run row attention
        r_attn_out = self.row_attn(r_attn_in, mask) # (B*(Fg+1), S, emb_dim)
        # Add residual and layer normalize
        res_r_attn_out = self.layer_norm_ra(r_attn_out + r_attn_in) # (B*(Fg+1), S, emb_dim)

        # Reshape and transpose back to original shape for feed-forward input
        ff_in = res_r_attn_out.reshape([B, Fg1, S, E]).transpose(1,2) # (B, S, Fg+1, emb_dim)
        # Run feed-forward
        ff_out = self.feed_forward(ff_in) # (B, S, Fg+1, emb_dim)
        # Add residual and layer normalize
        layer_out = self.layer_norm_ff(ff_out + ff_in) # (B, S, Fg+1, emb_dim)

        # Return layer output
        return layer_out # (B, S, Fg+1, emb_dim)


class Transformer(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Create all the layers for the transformer
        self.layers = nn.ModuleList([
            Layer(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, trans_in: torch.Tensor, test_size: int) -> torch.Tensor:

        # Iterate through the layers and return
        for layer in self.layers:
            trans_in = layer(trans_in, test_size)
        return trans_in


class EncoderX(nn.Module):

    def __init__(self, group_size: int, hidden_dim: int, embedded_dim: int):
        super().__init__()

        # Store the group size for later group rearrangement
        self.group_size = group_size

        # MLP to be applied to the feature groups to encode them
        self.mlp = nn.Sequential(
            nn.Linear(group_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Unsqueeze assuming batch size is 1. (B, S, F)
        if X.dim() == 1:
            X = X.view([1, X.shape[0], 1])

        if X.dim() == 2:
            X = X.view([1, X.shape[0], X.shape[1]])
        
        # Store initial shape
        B, S, F = X.shape

        # Append missing columns for feature group creation
        residue = F % self.group_size
        if residue != 0:
            X = torch.cat([X, torch.zeros([B, S, self.group_size - residue], dtype=torch.float32, device=X.device)], dim=2) # (B, S, F_pad)

            # Scale to keep overall group feature density given empty rows
            X[:, :,-self.group_size:] *= (self.group_size / residue) ** 0.5

        # Rearrange into feature groups
        Fg = X.shape[2] // self.group_size
        X = X.view(B, S, Fg, self.group_size) # (B, S, Fg, group_size)

        # Apply the MLP to bring tokens to embedded dimensions
        return self.mlp(X) # (B, S, Fg, emb_dim)
    

class EncoderY(nn.Module):

    def __init__(self, hidden_dim: int, embedded_dim: int):
        super().__init__()

        # MLP to be applied to the target plus mask to encode them
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, y: torch.Tensor, test_size: int) -> torch.Tensor:

        # Unsqueeze assuming batch size is 1. (B, train_size, 1)
        if y.dim() == 1:
            y = y.view([1, y.shape[0], 1])

        if y.dim() == 2:
            if y.shape[1] != 1:
                raise RuntimeError(
                    "Multiple target predictions are not supported. " \
                    f"Found target tensor of shape {y.shape}. " \
                    "If batch size is bigger than 1 the input is to be of shape (B, S, 1)."
                    )
            y = y.view([1, y.shape[0], y.shape[1]])

        # Store initial shape
        B, train_size, _ = y.shape

        # Append missing rows to y
        y = torch.cat([y, torch.zeros([B, test_size, 1], dtype=torch.float32, device=y.device)], dim=1) # (B, S, 1)

        # Create missing targets tensor
        mask = torch.zeros_like(y) # (B, S, 1)
        # Missing target is encoded as 1s
        mask[:,-test_size:,0] = 1

        # concatenate tensors
        y_plus_mask = torch.cat([y,mask], dim=2) # (B, S, 2)

        # Apply the MLP to bring target tokens to embedded dimensions and reshape to match features
        return self.mlp(y_plus_mask).view([B, train_size + test_size, 1, -1]) # (B, S, 1, emb_dim)


class Decoder(nn.Module):

    def __init__(self, embedded_dim: int, hidden_dim: int, n_buckets: int):
        super().__init__()

        # MLP to be applied to the embedded targets to turn them to logits
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_buckets)
        )

    def forward(self, trans_out: torch.Tensor, test_size: int) -> torch.Tensor:
        
        # Take the last tokens of the test rows (originally the unknown targets)
        target_tokens = trans_out[:,-test_size:,-1,:] # (B, test_size, emb_dim)

        # Apply the MLP to the tokens to obtain the raw logits
        return self.mlp(target_tokens) # (B, test_size, n_buckets)


class MyRegressorPFN(nn.Module):

    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()

        self.config = model_config if model_config is not None else ModelConfig()

        self.encoder_x = EncoderX(
            self.config.feature_group_size, 
            self.config.hidden_dimension_enc, 
            self.config.embedded_dimension
        )
        
        self.encoder_y = EncoderY(
            self.config.hidden_dimension_enc,
            self.config.embedded_dimension
        )

        self.transformer = Transformer(
            self.config
        )

        self.decoder = Decoder(
            self.config.embedded_dimension,
            self.config.hidden_dimension_dec,
            self.config.n_buckets
        )

        if self.config.n_thinking_rows != 0:
            self.thinking_token = nn.Parameter(torch.randn([self.config.embedded_dimension,], dtype=torch.float32, requires_grad=True))

        self.to(self.config.device)

    def forward(self):
        raise RuntimeError("This Module does not support forward passes, use fit() and predict() instead.")

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> MyRegressorPFN:

        # Store training tensors for inference
        self.X_train = X_train # (B?, train_size, F?)
        self.y_train = y_train # (B?, train_size, 1?)

        # Return itself for simple fit/predict concatenation
        return self

    def predict(self, X_test: torch.Tensor, output: str = 'logits') -> torch.Tensor:

        # Sanity check
        if not hasattr(self, 'X_train'):
            raise RuntimeError("Please call fit() before calling predict().")

        # Make sure your data are torch tensors
        self.X_train = torch.as_tensor(self.X_train, dtype=torch.float32, device=self.config.device) # (B?, train_size, F?)
        self.y_train = torch.as_tensor(self.y_train, dtype=torch.float32, device=self.config.device) # (B?, train_size, 1?)
        X_test       = torch.as_tensor(      X_test, dtype=torch.float32, device=self.config.device) # (B?, test_size?, F?)

        # Reshape assuming missing dimension means test_size is 1 (B?, test_size, F)
        if self.X_train.dim() == 2 and X_test.dim() == 1:
            X_test = X_test.view([1, -1])

        if self.X_train.dim() == 3 and X_test.dim() == 2:
            X_test = X_test.view([X_test.shape[0], 1, -1])
        
        # Get test size
        test_size = X_test.shape[-2]

        # Concatenate sets, error will occur if shapes missmatch
        X = torch.cat([self.X_train, X_test], dim = -2) # (B?, S = train_size + test_size, F)

        # Encode X and y
        emb_X = self.encoder_x(X)                        # (B, S, Fg, emb_dim)
        emb_y = self.encoder_y(self.y_train, test_size)  # (B, S,  1, emb_dim)

        # Concatenate vectors to obtain the input tokens
        emb = torch.cat([emb_X,emb_y], dim = 2) # (B, S, Fg+1, emb_dim)

        # Add thinking rows to get the final transformer input
        if self.config.n_thinking_rows != 0:
            B, _, Fg1, _ = emb.shape
            # Create thinking rows
            thinking_rows = torch.zeros(
                [B, self.config.n_thinking_rows, Fg1, self.config.embedded_dimension], 
                dtype=torch.float32, device=self.config.device
            )
            # Give all thinking tokens the learned thinking token value
            thinking_rows[:,:,:] = self.thinking_token
            # Concatenate
            trans_in = torch.cat([thinking_rows, emb], dim = 1)
        else:
            trans_in = emb

        # Send tokens through the transformer
        trans_out = self.transformer(trans_in, test_size)

        # Get logits from the decoder
        logits = self.decoder(trans_out, test_size)

        # Apply temperature if required
        if self.config.temperature != 1.0:
            logits /= self.config.temperature

        # Return corresponding output
        if output == 'logits':
            return logits

        if output == 'probs' or output == 'probabilities':
            return torch.softmax(logits, dim=-1)
        
        return logits
    