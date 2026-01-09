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
        return self.mlp(emb) # (B, S+T, Fg+1, emb_dim)
    

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
        B, S, Fg1, E = emb.shape # (B, S+T, Fg+1, emb_dim)

        # Reshape to create feature attention input
        f_attn_in = emb.reshape([B*S, Fg1, E]) # (B*(S+T), Fg+1, emb_dim)
        # Run feture attention
        f_attn_out = self.feature_attn(f_attn_in) # (B*(S+T), Fg+1, emb_dim)
        # Add residual and layer normalize
        res_f_attn_out = self.layer_norm_fa(f_attn_out + f_attn_in) # (B*(S+T), Fg+1, emb_dim)

        # Create attention mask for row attention (test attends to train, train attends to train)
        mask = torch.zeros([S, S], dtype=torch.float32, device=emb.device)
        mask[:,-test_size:] = float('-inf')

        # Reshape and transpose to create row attention input
        r_attn_in = res_f_attn_out.reshape([B, S, Fg1, E]).transpose(1,2).reshape([B*Fg1, S, E]) # (B*(Fg+1), S+T, emb_dim)
        # Run row attention
        r_attn_out = self.row_attn(r_attn_in, mask) # (B*(Fg+1), S+T, emb_dim)
        # Add residual and layer normalize
        res_r_attn_out = self.layer_norm_ra(r_attn_out + r_attn_in) # (B*(Fg+1), S+T, emb_dim)

        # Reshape and transpose back to original shape for feed-forward input
        ff_in = res_r_attn_out.reshape([B, Fg1, S, E]).transpose(1,2) # (B, S+T, Fg+1, emb_dim)
        # Run feed-forward
        ff_out = self.feed_forward(ff_in) # (B, S+T, Fg+1, emb_dim)
        # Add residual and layer normalize
        layer_out = self.layer_norm_ff(ff_out + ff_in) # (B, S+T, Fg+1, emb_dim)

        # Return layer output
        return layer_out # (B, S+T, Fg+1, emb_dim)


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
        return trans_in # (B, S+T, Fg+1, emb_dim)


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
        
        # Store initial shape
        B, S, F = X.shape # (B, S, F)

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

        # Store initial shape
        B, train_size, _ = y.shape # (B, train_size, 1)

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

    def reshape_concatenate_pre_encoder(
            device: str, 
            X_train: torch.Tensor, 
            y_train: torch.Tensor, 
            X_test: torch.Tensor
        ) -> list[torch.Tensor, torch.Tensor, int]:
        
        # Make sure your data are torch tensors
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device) # (B?, train_size, F?)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device) # (B?, train_size, 1?)
        X_test  = torch.as_tensor(X_test,  dtype=torch.float32, device=device) # (B?, test_size?, F?)

        # Reshape X_train tp match format (B, train_size, F)
        if X_train.dim() == 2:
            # For inference simplicity we will assume batch size to be 1
            # If you want multiple batches you will need to provide the full format
            X_train = X_train.unsqueeze(dim=0)
        elif X_train.dim() == 1:
            # For inference simplicity we will assume batch size is 1 and there is only 1 feature
            # A single training example case is rara so if that is the format it should be explicitly 
            # provided at least as a tensor of shape (1, F)
            X_train = X_train.unsqueeze(dim=0).unsqueeze(dim=-1)
        elif X_train.dim() != 3:
            # Sanity check
            raise RuntimeError(
                "Missmatch in shape of tensors found during preprocessing." \
                "X_train can only have 1, 2 or 3 dimensions."
            )

        # Store shape
        B, train_size, F = X_train.shape

        # Reshape y_train to match format (B, train_size, 1)
        if y_train.dim() == 3:
            # Dimensions must exactly match
            if y_train.shape[0] != B or y_train.shape[1] != train_size or y_train.shape[2] != 1:
                raise RuntimeError(
                    "Missmatch in shape of tensors found during preprocessing." \
                    "If X_train is of shape (B,train_size,F), a 3 dimensional y_train must exactly match the format (B, train_size, 1)."
                )
        elif y_train.dim() == 2:
            if B > 1:
                # Dimensions must be (B, train_size)
                if y_train.shape[0] != B or y_train.shape[1] != train_size:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, a 2 dimensional y_train must exactly match the format (B, train_size/1)."
                    )
                y_train = y_train.unsqueeze(dim=-1) # (B, train_size, 1)
            else: # train_size must match and the other dimension should be 1
                if y_train.shape[0] == train_size and y_train.shape[1] == 1:
                    y_train = y_train.unsqueeze(dim=0) # (B=1, train_size, 1)
                elif y_train.shape[0] == 1 and y_train.shape[1] == train_size:
                    y_train = y_train.unsqueeze(dim=-1) # (B=1, train_size, 1)
                else:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size 1, a 2 dimensional y_train must be of shape (train_size, 1) or (1, train_size)."
                    )
        elif y_train.dim() == 1:
            if B > 1:
                if y_train.shape[0] != B:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, y_train must match the batch size."
                    )
                # train_size can not be more than 1
                if train_size > 1:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size and train_size bigger than 1, y_train can not be one dimensional."
                    )
                # Assume format (B,)
                X_test = X_test.view([B, 1, 1]) # (B, train_size=1, 1)
            else: # train_size must match
                if y_train.shape[0] != train_size:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has train_size bigger than 1, y_train must contain the same train_size."
                    )
                # Assume format (train_size,)
                X_test = X_test.view([1, train_size, 1]) # (B=1, train_size, 1)
        else:
            # Sanity check
            raise RuntimeError(
                "Missmatch in shape of tensors found during preprocessing." \
                "y_train can only have 1, 2 or 3 dimensions."
            )

        # Reshape X_test tp match format (B, test_size, F)
        if X_test.dim() == 3:
            if X_test.shape[0] != B or X_test.shape[2] != F:
                raise RuntimeError(
                    "Missmatch in shape of tensors found during preprocessing." \
                    "If X_train is of shape (B,train_size,F), X_test with 3 dimensions must match B and F."
                )
        elif X_test.dim() == 2:
            if B > 1:
                # Batch size must match on both tensors
                if X_test.shape[0] != B:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, X_test must match the batch size."
                    )
                if F > 1: # Expect format (B, F) with test size 1
                    if X_test.shape[1] == F:
                        X_test = X_test.view([B, 1, F]) # (B, test_size=1, F)
                    else:
                        raise RuntimeError(
                            "Missmatch in shape of tensors found during preprocessing." \
                            "If X_train has a number of features bigger than 1, X_test must match the number of features."
                        )
                else: # Assume format (B, test_size) works if test_size is 1 as well
                    X_test = X_test.view([B, -1, 1]) # (B, test_size, F=1)
            else:
                if X_test.shape[1] == F: # Assume format (test_size, F) works if test_size is 1 and if F is 1 as well
                    X_test = X_test.view([1, -1, F])
                else:
                    if F > 1:
                        raise RuntimeError(
                            "Missmatch in shape of tensors found during preprocessing." \
                            "If X_train has a number of features bigger than 1, X_test must match the number of features."
                        )
                    if X_test.shape[0]>1:
                        raise RuntimeError(
                            "Missmatch in shape of tensors found during preprocessing." \
                            "X_train has 1 feature and batch size 1, if X_test is 2 dimensional at least one dimension must be 1."
                        )
                    # Assume shape (B=1, test_size)
                    X_test = X_test.view([1, -1, 1]) # (B=1, test_size, F=1)
        elif X_test.dim() == 1:
            if B > 1:
                # Batch size must match on both tensors
                if X_test.shape[0] != B:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1, X_test must match the batch size."
                    )
                # Feature length can not be more than 1
                if F > 1:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has batch size bigger than 1 and more than one feature, X_test can not be one dimensional."
                    )
                # Assume format (B,) and with test_size 1
                X_test = X_test.view([B, 1, 1]) # (B, test_size=1, F=1)
            elif F > 1:
                # Feature lenght must match
                if X_test.shape[0] != F:
                    raise RuntimeError(
                        "Missmatch in shape of tensors found during preprocessing." \
                        "If X_train has a number of features bigger than 1, X_test must match the number of features."
                    )
                # Assume format (F,) and with test_size 1
                X_test = X_test.view([1, 1, F]) # (B=1, test_size=1, F)
            else: # Assumbe format (test_size,)
                X_test = X_test.view([1, -1, 1]) # (B=1, test_size, F=1)
        else:
            # Sanity check
            raise RuntimeError(
                "Missmatch in shape of tensors found during preprocessing." \
                "X_test can only have 1, 2 or 3 dimensions."
            )

        # Get test size
        test_size = X_test.shape[1]

        # Concatenate sets
        X = torch.cat([X_train, X_test], dim = -2) # (B, S = train_size + test_size, F)

        # Return preprocessed tensors and test size
        return X, y_train, test_size

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

        # Preprocess tensors for encoders
        X, y, test_size = self.reshape_concatenate_pre_encoder(
            self.config.device, self.X_train, self.y_train, X_test
        )

        # Encode X and y
        emb_X = self.encoder_x(X)             # (B, S, Fg, emb_dim)
        emb_y = self.encoder_y(y, test_size)  # (B, S,  1, emb_dim)

        # Concatenate vectors to obtain the input tokens
        emb = torch.cat([emb_X,emb_y], dim = 2) # (B, S, Fg+1, emb_dim)

        # Add thinking rows to get the final transformer input
        if self.config.n_thinking_rows != 0:
            B, _, Fg1, _ = emb.shape
            # Create thinking rows # (B, T, Fg+1, emb_dim)
            thinking_rows = torch.zeros(
                [B, self.config.n_thinking_rows, Fg1, self.config.embedded_dimension], 
                dtype=torch.float32, device=self.config.device
            )
            # Give all thinking tokens the learned thinking token value
            thinking_rows[:,:,:] = self.thinking_token
            # Concatenate
            trans_in = torch.cat([thinking_rows, emb], dim = 1) # (B, S+T, Fg+1, emb_dim)
        else:
            trans_in = emb

        # Send tokens through the transformer
        trans_out = self.transformer(trans_in, test_size) # (B, S+T, Fg+1, emb_dim)

        # Get logits from the decoder
        logits = self.decoder(trans_out, test_size) # (B, test_size, n_buckets)

        # Apply temperature if required
        if self.config.temperature != 1.0:
            logits /= self.config.temperature # (B, test_size, n_buckets)

        # Return corresponding output
        if output == 'logits':
            return logits # (B, test_size, n_buckets)

        if output == 'probs' or output == 'probabilities':
            return torch.softmax(logits, dim=-1) # (B, test_size, n_buckets)
        
        raise RuntimeError(f"Unknown output type found: {output}")
    