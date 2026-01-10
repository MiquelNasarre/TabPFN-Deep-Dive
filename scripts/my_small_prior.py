from __future__ import annotations

import torch
import torch.nn as nn
import random

from dataclasses import dataclass, field

@dataclass
class PriorConfig:
    '''
    Dataclass that defines our prior through a set of hyper-parameters. To be 
    fed into the dataset generator. Can be used to configure how the causal graph 
    will be generated. For specifics check the class variables.
    '''

    # The probability of adding another layer to the MLP
    prob_next_layer: float = 0.5

    # The dropout of the connections in the MLP
    link_dropout_rate: float = 0.55

    # STD for the noise STD randomizer for each variable
    noise_multiplier: float = 0.1

    # Maximum amout of nodes each layer can have
    max_nodes_per_layer: int = 24

    # Possible non-linearities to be applied (sorry for the weirdness, otherwise ChatGPT says the list will be shared)
    non_linears: list = field(default_factory=lambda: [nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid])

    # Maximum number of training rows
    max_train_rows: int = 256

    # Number of test rows for every dataset
    n_test_rows: int = 128

    # Maximum number of features selected
    max_features: int = 24

    # Batch size
    batch_size: int = 4

    # Device to be used for tensors
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class RandomPriorMLP:
    '''
    Class that defines a randomly generated causal graph from our prior. 
    
    When initialized it creates an MLP with random number of layers, layer 
    sizes, dropout connections, non-linearities and node noise.

    When the forward pass is called it expects an input tensor of leading 
    dimension layer_sizes[0] and it will run the MLP and return all the 
    nodes of the MLP stacked into a tensor.
    '''

    # Stores the config for reference
    config: PriorConfig

    # Number of layers the MLP has
    num_layers: int

    # Number of nodes in each layer (including input and output)
    layer_sizes: list[int]

    # Matrices for the forward pass between layers with dropout (zeros)
    matrices: list[torch.Tensor]

    # Biases for the forward pass after linear projection
    biases: list[torch.Tensor]

    # Noise STD to be multiplied to each node noise addition
    node_noise: list[torch.Tensor]

    # Non linear functions used for each layer
    non_linears: list[nn.Module]

    def __init__(self, config: PriorConfig, min_nodes: int):

        # Store config for reference
        self.config = config

        # Create first two layer sizes
        self.layer_sizes = [random.randint(1, config.max_nodes_per_layer), random.randint(1, config.max_nodes_per_layer),]

        # Calculate total nodes
        total_nodes = self.layer_sizes[0] + self.layer_sizes[1]

        # Create the rest of the layer sizes
        while total_nodes < min_nodes or random.uniform(0, 1) < config.prob_next_layer:
            # Decide next layer size
            next_layer_size = random.randint(1, config.max_nodes_per_layer)
            # Append and add to total nodes
            self.layer_sizes.append(next_layer_size)
            total_nodes += next_layer_size

        # Store the total number of layers
        self.num_layers = len(self.layer_sizes)

        # Prepare lists
        self.matrices = []
        self.biases = []
        self.node_noise = []
        self.non_linears = []

        # Create data for each layer 
        for i in range(self.num_layers - 1):
            
            # Create the projection matrix
            self.matrices.append(torch.nn.init.kaiming_normal_(torch.zeros([self.layer_sizes[i], self.layer_sizes[i+1]], device=config.device)) / (1-config.link_dropout_rate)**0.5) # To keep variance stable

            # Apply dropout mask to the matrices
            dropout_mask = torch.bernoulli(torch.ones_like(self.matrices[-1]) * (1-config.link_dropout_rate))
            self.matrices[-1] = self.matrices[-1] * dropout_mask

            # Create the biases
            self.biases.append(torch.randn([self.layer_sizes[i+1],], device=config.device) / self.layer_sizes[i+1]**0.5)

            # Choose a non linearity for each layer
            non_linear_idx = random.randint(0, len(config.non_linears) - 1)
            self.non_linears.append(config.non_linears[non_linear_idx]())

            # Choose a random noise STD for each node
            self.node_noise.append(torch.abs(torch.randn([self.layer_sizes[i+1],], device=config.device)) * config.noise_multiplier)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Prepare a list to store all the layers and store input
        layers: list[torch.Tensor] = [input,]
        
        # Running variable to store previous layer
        prev_layer: torch.Tensor = input # (..., layer_sizes[0])
        # Iterate through the layers and store the outputs
        for i in range(self.num_layers - 1):
            # Get next layer through linear
            layer = prev_layer @ self.matrices[i] + self.biases[i] # (layer_sizes[i+1], layer_sizee[i]) @ (..., layer_sizes[i]) + (layer_sizes[i+1]) -> (..., layer_size[i+1])

            # Apply non-linearity
            layer = self.non_linears[i](layer) # non_lin((..., layer_sizes[i+1])) -> (..., non_lin(layer_sizes[i+1]))

            # Create random noise tensor and multiply by node noise
            noise = torch.randn_like(layer) * self.node_noise[i] # (..., layer_sizes[i+1]) * (layer_sizes[i+1]) -> (..., layer_sizes[i+1])

            # Add random noise to layer
            layer = layer + noise # (..., layer_sizes[i+1]) + (..., layer_sizes[i+1]) -> (..., layer_sizes[i+1])

            # Append layer to list and set to previous
            layers.append(layer)
            prev_layer = layer

        # Stack all the layers for easier output manipulation
        output = torch.cat(layers, dim = -1)

        # Return the stacked layers
        return output


def get_random_dataset(config: PriorConfig | None = None, return_mlp: bool = False):
    '''
    Given a prior configuration this function creates a random causal graph 
    via the class RandomPriorMLP and runs a forward pass on it, it selects 
    a random set of features and a target from the MLP nodes and reshapes it
    into a dataset.

    The dataset is then normalized and split into train and test data, and into
    features and target. It returns the X/y_train/test tensors and optionally 
    the RandomPriorMLP used to generate the data. 
    
    Args:
        config: Prior configuration used to generate the MLP and dataset.

        return_mlp: Whether the RandomPriorMLP used for generation is returned.
    '''

    # If no config provided default
    if config is None:
        config = PriorConfig()

    # Decide how mamy features the model will have
    num_features = random.randint(0, config.max_features)

    # Get a random MLP to draw features and target from
    mlp = RandomPriorMLP(config, min_nodes = num_features + 1)

    # Decide how many training samples you are going to draw
    training_rows = random.randint(1, config.max_train_rows)

    # Create an input tensor for the mlp with the training and test rows
    mlp_input = torch.randn([config.batch_size, training_rows + config.n_test_rows, mlp.layer_sizes[0]], device=config.device) # (B, S, layer_sizes[0])

    # Get the entire stack of nodes from the MLP
    mlp_out = mlp.forward(mlp_input)

    # Decide the node selection for the features/target of each batch
    # Generate random scores for each node
    scores = torch.rand(config.batch_size, mlp_out.shape[-1], device=mlp_out.device) # (B, total_nodes)
    # Then choose top-K indices per row (makes sampling non repeating)
    selection = scores.topk(k=num_features + 1, dim=-1).indices # (B, F+1)

    # Reshape and expand selection to gather from output shaped (B, S, total_nodes)
    selection = selection.unsqueeze(dim=1).expand([-1, mlp_out.shape[-2], -1]) # (B, S, F+1)

    # Gather selection from the output tensor
    dataset = torch.gather(mlp_out, dim=2, index=selection)  # (B, S, total_nodes) -> (B, S, F+1)
    
    # Normalize features and target before splitting
    mean = dataset.mean(dim=1, keepdim=True)                # (B, 1, F+1)
    std  = dataset.std(dim=1, keepdim=True).clamp(min=1e-5) # (B, 1, F+1)
    # Apply normalizations
    dataset = (dataset - mean) / std # broadcasts everything to (B, S, F+1)

    # Extract tensors from the dataset
    X_train = dataset[:, :training_rows, :-1].contiguous() # (B, train_size, F)
    y_train = dataset[:, :training_rows, -1:].contiguous() # (B, train_size, 1)
    X_test  = dataset[:, training_rows:, :-1].contiguous() # (B,  test_size, F)
    y_test  = dataset[:, training_rows:, -1:].contiguous() # (B,  test_size, 1)

    # Return tensors and optionally the generator MLP
    if return_mlp:
        return X_train, y_train, X_test, y_test, mlp
    else:
        return X_train, y_train, X_test, y_test


