from __future__ import annotations
from dataclasses import dataclass, field

# Basic imports
import torch
import math
import time
import sys

# Add project root to Python path
sys.path.append("C:/Users/PC/Desktop/TabPFN-Deep-Dive/")

# Import relevant classes for training loop
from scripts.my_small_prior import PriorConfig, get_random_dataset
from scripts.my_small_PFN import ModelConfig, MyRegressorPFN, BucketOps

@dataclass
class TrainConfig:
    '''
    Dataclass used to store the training configuration for a training run. 
    It defines training hyperparameters, load/save paths, and log events.

    The configuration parameters can also be added in the console as parse
    arguments, check the main path for more details.

    For details on the configuration variables check the variables directly.
    '''
    
    # Path to load the model from
    load_path: str | None = None

    # Path to save the model weights to
    save_path: str | None = "C:/Users/PC/Desktop/TabPFN-Deep-Dive/weights/my_PFN_weights.pth"

    # Model configuration
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())

    # Prior configuration
    prior_config: PriorConfig = field(default_factory=lambda: PriorConfig())

    # Gradient accumulation will be done until this batch size is reached
    effective_batch_size: int = 256

    # Initial learning rate for cosine decay
    initial_lr: float = 3e-4

    # Final learning rate for cosine decay
    final_lr: float = 3e-5

    # Weight decay (since datasets are unique and synthetic set to 0)
    weight_decay: float = 0.0

    # Momentum for SGD
    momentum: float = 0.9

    # Epochs that cosine decay will last
    cosine_epochs: int = 2000

    # Warmup epochs before cosine decay
    warmup_epochs: int = 100

    # If you want the training to finish automatically, set it here
    finish_at_epoch: int | None = None

    # Epochs between every log in console and save
    log_every: int = 1


def get_optimizer_scheduler(model: torch.nn.Module, config: TrainConfig):
    '''
    Function that returns the optimizer and scheduler for the training
    loop. The optimizer is AdamW and the scheduler does warmup, then 
    cosine decay and then a constant minimum learning rate value.
    '''

    # Optimizer AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay,
        betas=(config.momentum, 0.999),
        eps=1e-8,
    )

    # We implement lambda as a multiplier of initial_lr
    final_mult = config.final_lr / config.initial_lr

    # Create custom LR lambda function
    def lr_lambda(epoch):
        # If warming up return lineal
        if epoch < config.warmup_epochs:
            return (epoch+1) / config.warmup_epochs  # 0 -> 1
        
        # If done with decay return final
        if epoch >= config.cosine_epochs + config.warmup_epochs:
            return final_mult
        
        # progress in [0,1] after warmup
        t = (epoch - config.warmup_epochs) / config.cosine_epochs
        # cosine from 1 down to final_mult
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return final_mult + (1.0 - final_mult) * cosine

    # Create scheduler from custom lambda
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def training_loop(config: TrainConfig | None = None):
    '''
    Training loop function. Given a certain configuration it performs the training 
    loop as specified. 
    
    First it creates the model, loads the weights if a path is provided, and created 
    the optimizer, scheduler and the loss function.

    Then it starts the training, generates random datasets and feeds them to the model, 
    computes the loss and backpropagates. It updates weights when effective batch size
    is reached and logs running results into the console and saves as specified.
    '''

    # If no training config default
    if config is None:
        config = TrainConfig()

    # Initialize the model
    model = MyRegressorPFN(config.model_config)

    # If there is a path to load weights load them
    if config.load_path is not None:
        state = torch.load(config.load_path, map_location="cpu")
        model.load_state_dict(state)
    model.train()
    model.to(config.model_config.device)

    # Get the optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(model, config)
    optimizer.zero_grad()

    # Get the loss function
    loss_f = torch.nn.CrossEntropyLoss()

    # Setup params
    accum_batch_size: int = 0
    accum_loss_since_last_log: float = 0.0 
    epoch: int = 0
    accum_steps: int = max(1, config.effective_batch_size // config.prior_config.batch_size)
    # Reset time
    t0 = time.time()
    # Start the training loop!    
    while config.finish_at_epoch is None or epoch < config.finish_at_epoch:

        # Get the random datasets
        X_train, y_train, X_test, y_test = get_random_dataset(config.prior_config)

        # Add to the accumulated size
        accum_batch_size += X_train.shape[0]

        # Fit data to the model
        model.fit(X_train, y_train)

        # Get output logits from the model
        logits = model.predict(X_test, output='logits') # (B, test_size, n_buckets)

        # Sanity check
        assert logits.requires_grad, "logits has requires_grad=False; predict() is detaching or using no_grad()"

        # Obtain target labels through bucket discretization
        labels = torch.as_tensor(BucketOps.real_to_bucket(config.model_config.n_buckets, y_test), dtype=torch.long, device=logits.device) # (B, test_size)

        # Flatten everything for loss conputations
        flat_logits = logits.reshape(-1, config.model_config.n_buckets) # (B * test_size, n_buckets)
        flat_labels = labels.reshape(-1)                                # (B * test_size,)

        # Compute loss
        loss = loss_f(flat_logits, flat_labels) / accum_steps
        # Keep track
        accum_loss_since_last_log += loss.item()

        # Backpropagate!
        loss.backward()

        # Sanity check
        any_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert any_grad, "No finite gradients detected"

        # If effective size reached do gradient descent
        if accum_batch_size >= config.effective_batch_size:
            # Reset
            accum_batch_size = 0

            # Step that optimizer
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]

            # Reset gradients
            optimizer.zero_grad(set_to_none=True)

            # An effective batch means an epoch here
            scheduler.step()
            epoch += 1

            # If it is time to print do it
            if epoch % config.log_every == 0:
                # Check time
                t1 = time.time()
                elapsed = t1 - t0
                t0 = t1
                # Print to console
                print(f"Epoch {epoch:04} reached | LR: {lr:.6f} | Accumulated loss: {accum_loss_since_last_log:.4f} | Time elapsed {elapsed:.2f}s")
                accum_loss_since_last_log = 0.0

                # If there is a save path use it
                if config.save_path is not None:
                    torch.save(model.state_dict(), config.save_path)


if __name__ == "__main__":

    # Parse custom configurations
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--effective_batch_size", type=int)
    parser.add_argument("--initial_lr", type=float)
    parser.add_argument("--final_lr", type=float)
    parser.add_argument("--cosine_epochs", type=int)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--finish_at_epoch", type=int)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"])

    # Get args
    args = parser.parse_args()

    # Load model config
    config = TrainConfig()

    # Apply args if they exist
    for key, value in vars(args).items():
        if value is not None:
            if hasattr(config, key):
                setattr(config, key, value)
            elif key == "device":
                config.prior_config.device = value
                config.model_config.device = value
            else:
                raise ValueError(f"Unknown CLI argument: {key}")

    # Start the training!
    print("Starting training MyRegressorPFN on synthetic datasets\n")
    training_loop(config)
